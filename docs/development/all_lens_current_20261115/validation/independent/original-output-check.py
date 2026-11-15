"""Independent fixed-N catalog and continuous ideal-field audit; no transport."""
from copy import copy, deepcopy
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import tomllib
from unittest.mock import patch

import numpy as np
from scipy.constants import mu_0
from scipy.integrate import quad

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.simulation_modes import switch_mode

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
CURRENT = 'ideal_coil_current_a_at_100_percent'
ELECTRICAL = {CURRENT, 'winding_turn_count', 'winding_interval_turn_counts'}
ROUND = {'magnetic_lens_assembly', 'integrated_magnetic_lens_channel'}


def stripped(doc):
    result = deepcopy(doc)
    for part in result.get('parts', ()):
        for name in ELECTRICAL:
            part.pop(name, None)
        metadata = part.get('parameter_metadata', {})
        for name in ELECTRICAL:
            metadata.pop(name, None)
        if not metadata:
            part.pop('parameter_metadata', None)
    return result


def intervals(coil, owner):
    if all(f'{side}_yoke_start_local_z_mm' in owner for side in ('upper', 'lower')):
        inset = float(owner.get('mechanical_coil_axial_inset_mm', 0))
        return [(float(owner[f'{side}_yoke_start_local_z_mm'])+inset,
                 float(owner[f'{side}_yoke_end_local_z_mm'])-inset) for side in ('upper', 'lower')]
    return coil.get('material_intervals_mm', [(coil['local_start_z_mm'], coil['local_end_z_mm'])])


def fraction(value):
    # Decimal-text inputs are the persisted physical dimensions. This audit
    # avoids the migration helper's floating-point ratio and rounding code.
    return Fraction(str(value))


def exact_estimate(coil, occupied):
    radial = (fraction(coil['mechanical_outer_diameter_mm'])-fraction(coil['mechanical_inner_diameter_mm']))/2
    area = radial*sum((fraction(b)-fraction(a) for a, b in occupied), Fraction(0))
    quota = 1172*area/10440
    rounded = (2*quota.numerator+quota.denominator)//(2*quota.denominator)
    return max(len(occupied), rounded), float(area)


def state_for(catalog, column, recording):
    state = default_state(); catalog.apply(state, AssemblySelection('FEG', column, recording))
    switch_mode(state, 'unified')
    selected = catalog.default_selection()
    if (column, recording) == (selected.column, selected.recording):
        apply_operating_mode_pair(state, 'nano_probe', 'diffraction', column_name=column, recording_name=recording)
    apply_physical_layout_to_state(state); state.step_mm = .05
    return state


def independent_bz(z_mm, geometry):
    z = z_mm*.001
    total = 0.
    for row in geometry:
        before = z-row['start_m']; after = z-row['end_m']
        radial = quad(lambda radius: before/math.sqrt(radius*radius+before*before)
                      -after/math.sqrt(radius*radius+after*after), row['inner_m'], row['outer_m'],
                      epsabs=1e-13, epsrel=1e-13)[0]
        total += mu_0*row['ampere_turns']/(2*(row['outer_m']-row['inner_m'])*(row['end_m']-row['start_m']))*radial
    return total


def run():
    baseline = json.loads((PARENT/'before/baseline.json').read_text())
    checks = json.loads((PARENT/'field-checks.json').read_text())
    catalog = AssemblyCatalog(); states = {}; rows = []; selected = {}; arrays = {}
    for filename in baseline['files']:
        path = ROOT/filename
        document = tomllib.loads(path.read_text(encoding='utf-8'))
        before = tomllib.loads((PARENT/'before'/filename).read_text(encoding='utf-8'))
        assert stripped(document) == stripped(before), f'Non-electrical change: {filename}'
        by = {p['key']: p for p in document.get('parts', ())}
        for owner in by.values():
            if owner.get('mechanical_profile') not in ROUND:
                continue
            coils = [c for c in by.values() if c.get('mechanical_profile') == 'magnetic_excitation_coil'
                     and c.get('field_source_key') == owner['key']]
            assert len(coils) == 1
            coil = coils[0]; occupied = intervals(coil, owner)
            count, area = exact_estimate(coil, occupied)
            assert coil['winding_turn_count'] == count, (filename, owner['key'], count)
            regional = coil['winding_interval_turn_counts']
            assert sum(regional) == count and len(regional) == len(occupied)
            assert all(type(n) is int and n > 0 for n in regional)
            if len(regional) == 1:
                assert regional == [count]
            else:
                assert regional == [586, 586] and count == 1172
            assert 'winding_wire_diameter_mm' not in coil
            assert math.isfinite(owner[CURRENT]) and owner[CURRENT] >= 0
            assert coil['parameter_metadata']['winding_turn_count']['source_kind'] == 'estimated'
            assert owner['parameter_metadata'][CURRENT]['source_kind'] == 'user_defined'
            case = next(c for c in checks if c['file'] == filename and c['owner'] == owner['key'])
            case_key = (case['column'], case['recording'])
            if case_key not in states:
                states[case_key] = state_for(catalog, *case_key)
            state = states[case_key]
            controls = deepcopy(state.to_dict())
            captured = copy(state); captured.lenses = deepcopy(state.lenses)
            captured.round_lens_field_model = 'copper_coil'
            captured._runtime_lens_field_provider_cache = {}; captured._runtime_copper_coil_provider_cache = {}
            lens = next(l for l in captured.lenses if l.key == owner['key'])
            lens.enabled = True; lens.percent = 100.
            potential = capture_round_lens_potential(captured, owner['key']).potential
            assert len(potential.coils) == len(regional)
            expected_ni = [n*owner[CURRENT]*lens.polarity for n in regional]
            assert [c.ampere_turns for c in potential.coils] == expected_ni
            offset = float(state._resolved_assembly.part(owner['key']).start_z_mm)-float(owner['local_start_z_mm'])
            geometry = []
            for component, (start, end), ni in zip(potential.coils, occupied, expected_ni, strict=True):
                expected = [coil['mechanical_inner_diameter_mm']*.0005,
                            coil['mechanical_outer_diameter_mm']*.0005, (end-start)*.001,
                            (start+end+2*offset)*.0005]
                np.testing.assert_allclose([component.inner_radius_m, component.outer_radius_m,
                                           component.length_m, component.center_z_m], expected, rtol=0, atol=2e-15)
                assert component.quadrature_order == 12
                geometry.append(dict(inner_m=expected[0], outer_m=expected[1], start_m=(start+offset)*.001,
                                     end_m=(end+offset)*.001, ampere_turns=ni))
            witness = potential.provenance['winding_current_definition']['windings'][0]
            assert witness['wire_diameter_mm'] is None and witness['conductor_cross_section_mm2'] is None
            plane = case['matching_plane_z_mm']
            field = float(potential.sample([[0., 0., plane*.001]])['B_t'][0, 2])
            np.testing.assert_allclose(field, case['reference_target_bz_t'], rtol=2e-14, atol=0)
            row = dict(file=filename, owner=owner['key'], estimated_turns=count, cross_section_mm2=area,
                       interval_turns=regional, I100_a=owner[CURRENT], interval_reference_NI=expected_ni,
                       matching_plane_z_mm=plane, captured_reference_bz_t=field,
                       independent_reference_bz_t=independent_bz(plane, geometry),
                       potential_digest=potential.digest, input_sha256=sha256(path.read_bytes()).hexdigest())
            rows.append(row)
            unique = owner['key']+'_'+str(count)
            if unique not in selected:
                lower = min(c['start_m'] for c in geometry)*1000-100.
                upper = max(c['end_m'] for c in geometry)*1000+100.
                z = np.unique(np.append(np.linspace(lower, upper, 121), plane))
                actual = sum(c.axis_bz_t(z*.001) for c in potential.coils)
                reference = np.array([independent_bz(p, geometry) for p in z])
                arrays[unique+'_z_mm'] = z; arrays[unique+'_q12_t'] = actual; arrays[unique+'_independent_t'] = reference
                selected[unique] = dict(row, sampled_curve_points=len(z),
                    sampled_peak_normalized_q12_error=float(np.max(np.abs(actual-reference))/np.max(np.abs(reference))),
                    relative_matching_plane_q12_error=(field/row['independent_reference_bz_t']-1))
            assert state.to_dict() == controls
        print('Independent checks:', filename, flush=True)
    assert len(rows) == 78 and len({r['owner'] for r in rows}) == 20 and len(selected) == 21
    np.savez_compressed(OUT/'independent-curves.npz', **arrays)
    result = dict(scope='Independent raw-input, NI and continuous field audit; no source or propagation',
                  source_files=9, raw_owner_checks=78, distinct_lens_keys=20, distinct_key_geometry_curves=21,
                  exact_decimal_area_rounding_passed=True, nonelectrical_inputs_unchanged=True,
                  all_reference_NI_and_geometries_passed=True, wire_diameter_inferred=False,
                  maximum_curve_peak_normalized_q12_error=max(r['sampled_peak_normalized_q12_error'] for r in selected.values()),
                  maximum_matching_plane_q12_error=max(abs(r['relative_matching_plane_q12_error']) for r in selected.values()),
                  error_scope='Diagnostics of unchanged q12 quadrature against continuous ideal coils; not a tolerance change',
                  rows=rows, distinct_curve_results=selected)
    (OUT/'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'distinct_curve_results')}, indent=2))


def forbidden(*args, **kwargs):
    raise AssertionError('Independent field audit must not emit or transport particles')


if __name__ == '__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden):
        run()
