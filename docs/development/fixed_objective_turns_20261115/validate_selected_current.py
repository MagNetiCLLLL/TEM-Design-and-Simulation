"""Validate the selected specimen-plane current from the installed defaults.

Field-only: no source emission, transport, focusing or coherent propagation.
Historical alternative-calibration evidence is read without modification.
"""
from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar

from temsim.assembly_catalog import AssemblyCatalog
from temsim.coil_winding import OWNER_CURRENT_FIELD, INTERVAL_TURNS_FIELD
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.physics.copper_coil_field_provider import CopperCoilAssemblyPotential
from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.round_lens_models import round_lens_cache_identity
from temsim.simulation_modes import switch_mode

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/selected-objective-current-20261115'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def remove_current(state):
    result = copy(state)
    result._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(p, data={k: v for k, v in p.data.items() if k != OWNER_CURRENT_FIELD})
        if p.key == 'objective_lens' else p for p in state._resolved_assembly.parts))
    result._runtime_copper_coil_provider_cache = {}
    result._runtime_lens_field_provider_cache = {}
    return result


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = AssemblyCatalog(); selection = catalog.default_selection(); state = default_state()
    catalog.apply(state, selection); switch_mode(state, 'unified')
    apply_operating_mode_pair(state, 'nano_probe', 'diffraction', column_name=selection.column,
                             recording_name=selection.recording)
    apply_physical_layout_to_state(state); state.step_mm = .05
    controls = deepcopy(state.to_dict())
    previous = json.loads((ROOT / 'docs/development/coil_inventory_20261115/validation/standard-startup-state.json').read_text())
    assert json.loads(json.dumps(controls)) == previous, 'Optical/source/numerical controls changed'
    owner = state._resolved_assembly.part('objective_lens')
    coil = state._resolved_assembly.part('objective_lens_excitation_coil')
    assert coil.data[INTERVAL_TURNS_FIELD] == (586, 586) and coil.data['winding_turn_count'] == 1172
    assert 'winding_wire_diameter_mm' not in coil.data
    historical = json.loads((HERE / 'validation/calibration.json').read_text())
    selected = next(row for row in historical['candidates'] if row['condition'] == 'specimen_upper_surface')
    assert owner.data[OWNER_CURRENT_FIELD] == selected['current_at_100_percent_a']
    lens = next(l for l in state.lenses if l.key == 'objective_lens')
    plane = float(state.sample.upper_surface_z_mm)
    gaussian = resolve_runtime_lens_field_provider(state, lens.key, lens)
    target = float(gaussian.magnetic_field_t(plane))
    ideal = copy(state); ideal.lenses = deepcopy(state.lenses)
    ideal.round_lens_field_model = 'copper_coil'
    ideal._runtime_copper_coil_provider_cache = {}; ideal._runtime_lens_field_provider_cache = {}
    potential = capture_round_lens_potential(ideal, lens.key).potential
    current = potential.provenance['winding_current_definition']['current_a']
    assert current == selected['current_a']
    assert [c.ampere_turns for c in potential.coils] == [586 * current] * 2
    assert all(c.quadrature_order == 12 for c in potential.coils)
    for actual, old in zip(potential.coils, selected['potential']['metadata']['coils'], strict=True):
        assert json.loads(json.dumps(actual.to_dict())) == old, 'The previously selected field coefficients changed'
    field = lambda z: sum(c.axis_bz_t(np.asarray(z) * .001) for c in potential.coils)
    observed = float(potential.sample([[0., 0., plane * .001]])['B_t'][0, 2])
    np.testing.assert_allclose(observed, target, rtol=2e-14, atol=0)
    z = np.linspace(1300., 1900., 6001)
    curves = {'z_mm': z, 'original_gaussian_bz_t': gaussian.magnetic_field_t(z),
              'selected_ideal_bz_t': field(z)}
    peaks = []
    for bounds in ((1400., 1550.), (1650., 1800.)):
        fit = minimize_scalar(lambda value: -float(field(value)), bounds=bounds, method='bounded',
                              options={'xatol': 1e-9})
        peaks.append({'z_mm': float(fit.x), 'bz_t': -float(fit.fun)})
    # Other ideal sources retain their existing definitions; no all-column model
    # selection or uncalibrated-current substitution is installed at startup.
    legacy = remove_current(ideal)
    other_checks = []
    for other in ideal.lenses:
        if not other.enabled or other.key == lens.key:
            continue
        before = capture_round_lens_potential(legacy, other.key).potential
        after = capture_round_lens_potential(ideal, other.key).potential
        assert before.to_dict() == after.to_dict()
        other_checks.append(other.key)
    assert round_lens_cache_identity(ideal) != round_lens_cache_identity(legacy)
    old_potential = capture_round_lens_potential(legacy, lens.key).potential
    assert old_potential.digest != potential.digest
    restored = CopperCoilAssemblyPotential.from_dict(potential.to_dict())
    np.testing.assert_array_equal(restored.sample([[0., 0., plane * .001]])['B_t'],
                                  potential.sample([[0., 0., plane * .001]])['B_t'])
    assert state.to_dict() == controls and state.round_lens_field_model == 'gaussian'
    ideal_controls = deepcopy(ideal.to_dict()); ideal_controls.pop('round_lens_field_model')
    assert ideal_controls == controls, 'Detached validation changed more than the field selection'
    np.savez_compressed(OUT / 'selected-field-curves.npz', **curves)
    save('standard-startup-state.json', controls)
    save('ideal-field-check-state.json', ideal.to_dict())
    save('selected-potential.json', potential.to_dict())
    config = ROOT / 'configs/instruments/column/C3_ProbeCorrector.toml'
    (OUT / 'C3_ProbeCorrector.toml').write_bytes(config.read_bytes())
    save('selected-current.json', {
        'status': 'selected_current_saved_and_field_verified',
        'scope': 'Objective contribution only; no source, particle, focus or coherent calculation',
        'matching_condition': 'original_specimen_upper_surface_bz',
        'fixed_interval_turns': [586, 586], 'total_series_turns': 1172,
        'upper_turn_source': historical['upper_turn_source'], 'lower_turn_source': historical['lower_turn_source'],
        'source_photo_sha256': historical['source_photo_sha256'],
        'current_at_100_percent_a': owner.data[OWNER_CURRENT_FIELD], 'excitation_percent': lens.percent,
        'current_a': current, 'interval_ampere_turns': [c.ampere_turns for c in potential.coils],
        'specimen_upper_surface_z_mm': plane, 'target_original_gaussian_bz_t': target,
        'selected_ideal_bz_t': observed, 'relative_matching_error': (observed-target)/target,
        'matching_check_rtol': 2e-14, 'matching_check_atol_t': 0., 'quadrature_order': 12,
        'ideal_peaks': peaks, 'wire_diameter_mm': None, 'pole_piece_magnetism': False,
        'thermal_or_current_rating_enforced': False, 'startup_field_model': state.round_lens_field_model,
        'validation_field_model': ideal.round_lens_field_model, 'other_ideal_sources_unchanged': other_checks,
        'full_optical_source_numerical_state_unchanged': True, 'cache_identity_changed': True,
        'previous_selected_coefficients_exactly_preserved': True, 'potential_roundtrip_exact': True,
        'config_sha256': sha256(config.read_bytes()).hexdigest(), 'potential_digest': potential.digest,
    })
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax in axes:
        ax.plot(z, curves['original_gaussian_bz_t'], label='Original Gaussian')
        ax.plot(z, curves['selected_ideal_bz_t'], label='Ideal coils: 586 + 586 turns')
        ax.axvline(plane, color='black', ls=':', label='Original specimen upper surface')
        ax.set(xlabel='Column z (mm)', ylabel='Objective contribution Bz (T)')
        ax.grid(alpha=.25)
    axes[0].set(title=f'Selected common current: {current:.6f} A', xlim=(1300, 1900))
    axes[1].set(title='Same surface field; different axial profile', xlim=(plane-40, plane+40), ylim=(0, 2.1))
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / 'selected-field-comparison.png', dpi=180); plt.close(fig)
    print(json.dumps({'current_a': current, 'I100_a': owner.data[OWNER_CURRENT_FIELD],
                      'surface_bz_t': observed, 'target_bz_t': target, 'peaks': peaks,
                      'unchanged_other_sources': len(other_checks), 'controls_unchanged': True}, indent=2))


def forbidden(*args, **kwargs):
    raise AssertionError('Field calibration must not emit or transport particles')


if __name__ == '__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden):
        run()
