"""Explicit catalog winding migration and field-only validation.

The saved pre-change baseline is never overwritten. --apply requires those
exact source bytes; the default action only validates already installed inputs.
No source emission, transport, focus adjustment or coherent run is performed.
"""
from argparse import ArgumentParser
from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.cad_round_winding import round_winding_envelope
from temsim.coil_current_calibration import calibrate_round_lens_current
from temsim.coil_turn_estimation import estimate_winding_turns
from temsim.coil_winding import OWNER_CURRENT_FIELD, INTERVAL_TURNS_FIELD
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.round_lens_models import round_lens_cache_identity
from temsim.simulation_modes import switch_mode

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/all-lens-current-20261115'
CURATED = Path(__file__).resolve().parent / 'validation'
ROUND = {'magnetic_lens_assembly', 'integrated_magnetic_lens_channel'}
ELECTRICAL = {'winding_turn_count', INTERVAL_TURNS_FIELD, OWNER_CURRENT_FIELD}


def input_path(name):
    for base in (OUT, CURATED):
        path = base / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'Preserved validation input is missing: {name}')


def save(name, value):
    destination = OUT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def startup(catalog, column, recording):
    state = default_state(); catalog.apply(state, AssemblySelection('FEG', column, recording))
    switch_mode(state, 'unified')
    selection = catalog.default_selection()
    if (column, recording) == (selection.column, selection.recording):
        apply_operating_mode_pair(state, 'nano_probe', 'diffraction', column_name=column, recording_name=recording)
    apply_physical_layout_to_state(state); state.step_mm = .05
    return state


def detached(state, updates=None, *, ideal=False):
    result = copy(state); result.lenses = deepcopy(state.lenses)
    result._runtime_lens_field_provider_cache = {}; result._runtime_copper_coil_provider_cache = {}
    result.round_lens_field_model = 'copper_coil' if ideal else 'gaussian'
    if updates:
        result._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
            replace(p, data={**p.data, **updates[p.key]}) if p.key in updates else p
            for p in state._resolved_assembly.parts))
    return result


def native(state, key):
    lens = next(l for l in state.lenses if l.key == key)
    return state.condenser_system[key] if key in CONDENSER_LENS_KEYS else lens


def build_plan():
    baseline = json.loads(input_path('before/baseline.json').read_text())
    catalog = AssemblyCatalog(); selection = catalog.default_selection()
    reference_file = ROOT / 'configs/instruments/column/C3_ProbeCorrector.toml'
    reference_doc = module_manifest.read_document(reference_file)
    reference_parts = {p['key']: p for p in reference_doc['parts']}
    reference_coil = reference_parts['objective_lens_excitation_coil']
    reference = round_winding_envelope(reference_coil, reference_parts)
    assert reference_coil['winding_turn_count'] == 1172
    entries = []
    for filename in baseline['files']:
        path = ROOT / filename
        assert sha256(path.read_bytes()).hexdigest() == baseline['files'][filename], 'Pre-change input bytes changed'
        document = module_manifest.read_document(path)
        by_key = {p['key']: p for p in document['parts']}
        column = next((o.name for o in catalog.columns if Path('configs/instruments')/o.file == Path(filename)), selection.column)
        recording = 'Energy Filter' if filename.endswith('projector_stack.toml') else selection.recording
        state = startup(catalog, column, recording)
        for coil in document['parts']:
            owner = by_key.get(coil.get('field_source_key'), {})
            if coil.get('mechanical_profile') != 'magnetic_excitation_coil' or owner.get('mechanical_profile') not in ROUND:
                continue
            envelope = round_winding_envelope(coil, by_key)
            estimate = estimate_winding_turns(envelope, reference, reference_coil['winding_turn_count'])
            counts = {'winding_turn_count': estimate['winding_turn_count'],
                      INTERVAL_TURNS_FIELD: estimate[INTERVAL_TURNS_FIELD]}
            candidate = detached(state, {coil['key']: counts})
            objective = owner['key'] == 'objective_lens'
            plane = float(state.sample.upper_surface_z_mm) if objective else None
            calibration = calibrate_round_lens_current(candidate, owner['key'], matching_plane_z_mm=plane)
            current = owner.get(OWNER_CURRENT_FIELD, calibration['current_at_100_percent_a'])
            if OWNER_CURRENT_FIELD in owner:
                np.testing.assert_allclose(current, calibration['current_at_100_percent_a'], rtol=2e-14, atol=0)
            lens = next(l for l in state.lenses if l.key == owner['key'])
            entries.append(dict(file=filename, owner=owner['key'], coil=coil['key'], label=owner['name'],
                column=column, recording=recording, envelope=envelope, estimate=estimate,
                existing_reference_preserved=OWNER_CURRENT_FIELD in owner,
                current_at_100_percent_a=current, calibration=calibration,
                plane_offset_from_native_z_mm=calibration['matching_plane_z_mm']-lens.z_mm))
        print('Prepared', filename, flush=True)
    assert len(entries) == 78 and len({e['owner'] for e in entries}) == 20
    save('migration-plan.json', dict(reference_file=str(reference_file.relative_to(ROOT)), reference_envelope=reference,
                                   reference_turn_count=1172, method='reference_winding_rz_area_ratio_v1', entries=entries))
    return entries


def apply_plan(entries):
    import tomllib
    from temsim.coil_reference_metadata import validate_coil_references
    from temsim.coil_winding import validate_winding_definition
    baseline = json.loads(input_path('before/baseline.json').read_text())
    staged = {}
    for filename, digest in baseline['files'].items():
        path = ROOT / filename
        assert sha256(path.read_bytes()).hexdigest() == digest, 'Refusing to overwrite changed physical inputs'
        doc = module_manifest.read_document(path); by_key = {p['key']: p for p in doc['parts']}
        updates = {}
        for row in (e for e in entries if e['file'] == filename):
            if row['existing_reference_preserved']:
                continue
            estimate, calibration = row['estimate'], row['calibration']
            note = (f"User-authorized engineering estimate on 2026-11-15: N=round_half_up(1172 * "
                    f"{estimate['cross_section_mm2']:.17g} / 10440)={estimate['winding_turn_count']}. "
                    "Use original winding-pack R-Z cross-sectional area, assuming equal turn density to the "
                    "Objective reference (upper 586 photo estimate, lower 586 symmetry assumption). "
                    "Not a measured or OEM turn count. Counts are fixed after this explicit estimate; "
                    "wire diameter remains unknown. Geometry edits do not re-estimate turns automatically.")
            coil_meta = deepcopy(by_key[row['coil']].get('parameter_metadata', {}))
            for field in ('winding_turn_count', INTERVAL_TURNS_FIELD):
                updates['parts', row['coil'], field] = estimate[field]
                coil_meta[field] = dict(source_kind='estimated', source_note=note)
            updates['parts', row['coil'], 'parameter_metadata'] = coil_meta
            owner_meta = deepcopy(by_key[row['owner']].get('parameter_metadata', {}))
            condition = 'original specimen upper surface' if row['owner'] == 'objective_lens' else 'original native Gaussian absolute-peak plane'
            owner_meta[OWNER_CURRENT_FIELD] = dict(source_kind='user_defined', source_note=(
                f"Current-only ideal-vacuum calibration on 2026-11-15 at the {condition}, "
                f"offset {row['plane_offset_from_native_z_mm']:.17g} mm from the native optical z reference. "
                f"At explicit enabled 100% reference excitation the original signed Bz is "
                f"{calibration['reference_target_bz_t']:.17g} T. Preserve the saved N, envelope, "
                "working excitation and polarity. No pole-piece response, resistance, thermal/current limit "
                "or focus equivalence is asserted. Subsequent N or current edits do not auto-refit this match. "
                "The historical 3000 A-turn fallback is retained but inactive for this explicit-current source."))
            updates['parts', row['owner'], OWNER_CURRENT_FIELD] = row['current_at_100_percent_a']
            updates['parts', row['owner'], 'parameter_metadata'] = owner_meta
        text = path.read_bytes().decode('utf-8')
        subassembly = filename.endswith('projector_stack.toml')
        staged[filename] = module_manifest.stage_manifest_text(text, updates, source_path=path, validate=not subassembly)
        if subassembly:
            # An independent part file is validated through its composed host,
            # not as a standalone column with a drift-vacuum boundary.
            candidate = tomllib.loads(staged[filename])
            validate_coil_references(candidate['parts'])
            for part in candidate['parts']:
                validate_winding_definition(part)
    # Prepare and schema-check all definitions before writing any of them.
    for filename, text in staged.items():
        assert sha256((ROOT / filename).read_bytes()).hexdigest() == baseline['files'][filename]
        (ROOT / filename).write_bytes(text.encode('utf-8'))
    print('Installed explicit winding/current inputs in', len(staged), 'definitions', flush=True)


def without_electrical(document):
    document = deepcopy(document)
    for part in document.get('parts', ()):
        for field in ELECTRICAL:
            part.pop(field, None)
        metadata = part.get('parameter_metadata', {})
        for field in ELECTRICAL:
            metadata.pop(field, None)
        if not metadata:
            part.pop('parameter_metadata', None)
    return document


def validate(entries):
    import tomllib
    baseline = json.loads(input_path('before/baseline.json').read_text())
    original_parts = {}
    for filename in baseline['files']:
        before = tomllib.loads(input_path('before/'+filename).read_text(encoding='utf-8'))
        after = tomllib.loads((ROOT / filename).read_text(encoding='utf-8'))
        assert without_electrical(before) == without_electrical(after), 'Non-electrical parameters changed'
        original_parts[filename] = {p['key']:p for p in before['parts']}
    catalog = AssemblyCatalog(); checks = []; curves = {}; plot_rows = {}
    for case in baseline['assemblies']:
        state = startup(catalog, case['column'], case['recording'])
        controls = deepcopy(state.to_dict())
        assert json.loads(json.dumps(controls)) == case['state'], 'Optical/source/numerical controls changed'
        for lens in state.lenses:
            owner = state._resolved_assembly.part(lens.key) if lens.key in {p.key for p in state._resolved_assembly.parts} else None
            if owner is None or owner.data.get('mechanical_profile') not in ROUND:
                continue
            filename = ('configs/instruments/'+next(o.file for o in catalog.columns if o.name == case['column'])
                        if owner.module_key.startswith('column:') else
                        'configs/subassemblies/projector_stack.toml' if case['recording'] == 'Energy Filter' else
                        'configs/instruments/project_and_recording_system/NoEnergyFilter.toml')
            row = next(e for e in entries if e['file'] == filename and e['owner'] == lens.key)
            winding = state._resolved_assembly.part(row['coil'])
            assert winding.data['winding_turn_count'] == row['estimate']['winding_turn_count']
            assert tuple(winding.data[INTERVAL_TURNS_FIELD]) == tuple(row['estimate'][INTERVAL_TURNS_FIELD])
            assert 'winding_wire_diameter_mm' not in winding.data
            assert owner.data[OWNER_CURRENT_FIELD] == row['current_at_100_percent_a']
            plane = lens.z_mm + row['plane_offset_from_native_z_mm']
            fitted = calibrate_round_lens_current(state, lens.key, matching_plane_z_mm=plane)
            reference = detached(state, ideal=True)
            ref_lens = next(l for l in reference.lenses if l.key == lens.key)
            ref_lens.enabled = True; ref_lens.percent = 100.
            potential = capture_round_lens_potential(reference, lens.key).potential
            observed = float(potential.sample([[0., 0., plane*.001]])['B_t'][0, 2])
            target = fitted['reference_target_bz_t']
            np.testing.assert_allclose(observed, target, rtol=2e-14, atol=0)
            assert all(c.quadrature_order == 12 for c in potential.coils)
            sign = ref_lens.polarity
            assert [c.ampere_turns for c in potential.coils] == [n*row['current_at_100_percent_a']*sign for n in winding.data[INTERVAL_TURNS_FIELD]]
            actual_current = row['current_at_100_percent_a']*(lens.percent*.01)*lens.polarity if lens.enabled else 0.
            check = dict(column=case['column'], recording=case['recording'], owner=lens.key, file=filename,
                turns=winding.data['winding_turn_count'], interval_turns=list(winding.data[INTERVAL_TURNS_FIELD]),
                current_a=actual_current, current_at_100_percent_a=row['current_at_100_percent_a'],
                excitation_percent=lens.percent, enabled=lens.enabled, polarity=lens.polarity,
                matching_plane_z_mm=plane, reference_target_bz_t=target, selected_reference_bz_t=observed,
                relative_error=(observed-target)/target, potential_digest=potential.digest)
            checks.append(check)
            # One curve per logical lens/area, plus all original full input controls.
            curve_key = lens.key+'_'+str(winding.data['winding_turn_count'])
            if curve_key not in plot_rows:
                gaussian_state = detached(reference)
                gaussian = resolve_runtime_lens_field_provider(gaussian_state, lens.key, native(gaussian_state, lens.key))
                span = max(100., sum(c.length_m for c in potential.coils)*2000.)
                z = np.linspace(plane-span, plane+span, 1201)
                curves[curve_key+'_z_mm'] = z
                curves[curve_key+'_gaussian_t'] = gaussian.magnetic_field_t(z)
                curves[curve_key+'_ideal_t'] = sum(c.axis_bz_t(z*.001) for c in potential.coils)
                plot_rows[curve_key] = dict(check, label=row['label'])
        previous_ideal = detached(state, ideal=True)
        replacements = []
        for part in state._resolved_assembly.parts:
            source = next((row for row in entries if row['owner']==part.key or row['coil']==part.key
                           if (row['column']==case['column'] and part.module_key.startswith('column:'))
                           or (not part.module_key.startswith('column:') and row['recording']==case['recording'])),None)
            if source is not None:
                data = dict(part.data); old = original_parts[source['file']][part.key]
                for field in ELECTRICAL:
                    if field in old:
                        data[field] = old[field]
                    else:
                        data.pop(field,None)
                part = replace(part,data=data)
            replacements.append(part)
        previous_ideal._resolved_assembly = replace(state._resolved_assembly,parts=tuple(replacements))
        assert round_lens_cache_identity(detached(state,ideal=True)) != round_lens_cache_identity(previous_ideal)
        assert state.to_dict() == controls
        print('Verified', case['column'], '/', case['recording'], flush=True)
    assert len(checks) >= 78
    save('field-checks.json', checks); save('plot-index.json', plot_rows)
    np.savez_compressed(OUT/'all-field-curves.npz', **curves)
    save('validation-summary.json', dict(status='field_only_validation_passed', source_definitions=9,
        source_owner_instances=78, unique_round_lens_keys=20, assembly_variants=14, field_checks=len(checks),
        maximum_relative_matching_error=max(abs(c['relative_error']) for c in checks),
        rtol=2e-14, atol_t=0, quadrature_order=12, complete_controls_unchanged=True,
        nonelectrical_parameters_unchanged=True, wire_diameters_inferred=False,
        ideal_cache_identities_changed_for_all_assemblies=True,
        source_or_transport_executed=False, coherent_executed=False, focus_adjusted=False,
        startup_field_model='gaussian'))
    print('Field checks passed:', len(checks), flush=True)


def forbidden(*args, **kwargs):
    raise AssertionError('Catalog field calibration must not emit or transport particles')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Install the explicit migration against saved pre-change hashes')
    arguments = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if arguments.apply:
        entries = build_plan(); apply_plan(entries)
    else:
        entries = json.loads(input_path('migration-plan.json').read_text())['entries']
    validate(entries)


if __name__ == '__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden):
        main()
