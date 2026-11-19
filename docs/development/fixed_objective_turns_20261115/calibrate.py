"""Two detached current calibrations with fixed 586+586 Objective turns.

Reconstruct historical alternatives without editing the selected defaults or
overwriting the earlier evidence captured before the user's choice.
"""
from copy import copy,deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar

from temsim.assembly_catalog import AssemblyCatalog
from temsim.coil_winding import OWNER_CURRENT_FIELD,INTERVAL_TURNS_FIELD
from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.simulation_modes import switch_mode


ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/fixed-upper-turns-20261115-alternatives-replay'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    catalog=AssemblyCatalog();selection=catalog.default_selection();state=default_state()
    catalog.apply(state,selection);switch_mode(state,'unified')
    apply_operating_mode_pair(state,'nano_probe','diffraction',column_name=selection.column,recording_name=selection.recording)
    from temsim.column.state_layout import apply_physical_layout_to_state
    apply_physical_layout_to_state(state);state.step_mm=.05
    baseline=state.to_dict()
    previous=json.loads((ROOT/'docs/development/coil_inventory_20261115/validation/standard-startup-state.json').read_text())
    assert json.loads(json.dumps(baseline))==previous,'Existing optical/source controls changed'
    part=state._resolved_assembly.part('objective_lens_excitation_coil')
    assert part.data['winding_turn_count']==1172 and part.data[INTERVAL_TURNS_FIELD]==(586,586)
    assert 'winding_wire_diameter_mm' not in part.data
    saved_current=state._resolved_assembly.part('objective_lens').data.get(OWNER_CURRENT_FIELD)
    lens=next(l for l in state.lenses if l.key=='objective_lens')
    gaussian=resolve_runtime_lens_field_provider(state,lens.key,lens)
    plane=float(state.sample.upper_surface_z_mm)
    target=float(gaussian.magnetic_field_t(plane));percent=float(lens.percent)
    def capture(actual):
        detached=copy(state);detached.lenses=deepcopy(state.lenses)
        detached.round_lens_field_model='copper_coil'
        detached._runtime_copper_coil_provider_cache={};detached._runtime_lens_field_provider_cache={}
        detached._resolved_assembly=replace(state._resolved_assembly,parts=tuple(
            replace(p,data={**p.data,OWNER_CURRENT_FIELD:actual/(percent*.01)})
            if p.key=='objective_lens' else p for p in state._resolved_assembly.parts))
        return capture_round_lens_potential(detached,'objective_lens').potential
    unit=capture(1.)
    unit_field=lambda z:sum(c.axis_bz_t(np.asarray(z)*.001) for c in unit.coils)
    peaks=[]
    for a,b in ((1400.,1550.),(1650.,1800.)):
        fit=minimize_scalar(lambda z:-float(unit_field(z)),bounds=(a,b),method='bounded',options={'xatol':1e-9})
        peaks.append(dict(z_mm=float(fit.x),bz_t_per_a=-float(fit.fun)))
    peak=max(peaks,key=lambda p:p['bz_t_per_a'])
    old_peak=float(gaussian.magnetic_field_t(1599.2))
    conditions={'global_peak':old_peak/peak['bz_t_per_a'],'specimen_upper_surface':target/float(unit_field(plane))}
    curves={'z_mm':np.linspace(1300,1900,6001)}
    curves['original_gaussian_bz_t']=gaussian.magnetic_field_t(curves['z_mm'])
    records=[]
    for name,current in conditions.items():
        potential=capture(current)
        actual=potential.provenance['winding_current_definition']['current_a']
        np.testing.assert_allclose([c.ampere_turns for c in potential.coils],[586*actual]*2,rtol=0,atol=0)
        field=lambda z:sum(c.axis_bz_t(np.asarray(z)*.001) for c in potential.coils)
        record=dict(condition=name,current_a=actual,current_at_100_percent_a=current/(percent*.01),
            upper_turns=586,lower_turns=586,total_series_turns=1172,
            region_ampere_turns=[c.ampere_turns for c in potential.coils],
            specimen_upper_surface_z_mm=plane,specimen_plane_bz_t=float(field(plane)),
            global_peak_bz_t=float(field(peak['z_mm'])),global_peak_z_mm=peak['z_mm'],
            target_original_gaussian_peak_t=old_peak,target_original_specimen_bz_t=target,
            wire_diameter_mm=None,current_installed_by_this_driver=False,potential=potential.to_dict())
        if name=='global_peak':np.testing.assert_allclose(record['global_peak_bz_t'],old_peak,rtol=2e-14)
        else:np.testing.assert_allclose(record['specimen_plane_bz_t'],target,rtol=2e-14)
        records.append(record);curves[name+'_bz_t']=field(curves['z_mm'])
    assert state.to_dict()==baseline
    assert state._resolved_assembly.part('objective_lens').data.get(OWNER_CURRENT_FIELD)==saved_current
    np.savez_compressed(OUT/'fixed-turn-curves.npz',**curves)
    source=Path(r'C:/Users/54019/Documents/WeChat Files/wxid_ajm9pyn349ya22/FileStorage/Temp/8cadeb709d4d8252aee8a74cae607f7.jpg')
    save('calibration.json',dict(status='historical_alternatives_reconstructed',
        upper_turn_source='Adopted photographic estimate586; approximate image count580-590; user statesonecircleoneturn',
        lower_turn_source='User-authorized symmetric586 assumption, not independently measured',
        source_photo_sha256=sha256(source.read_bytes()).hexdigest(),source_photo_name=source.name,
        fixed_default_turn_counts=[586,586],original_optical_source_state_unchanged=True,
        default_field_model=state.round_lens_field_model,current_installed_by_this_driver=False,
        loaded_default_reference_current_a=saved_current,
        current_density_or_thermal_limit_enforced=False,scope='Ideal field-only, no material/particle/coherent run',
        unit_current_peaks=peaks,candidates=records))
    for r in records:print(r['condition'],r['current_a'],r['current_at_100_percent_a'],r['global_peak_bz_t'],r['specimen_plane_bz_t'])


def forbidden(*args,**kwargs):raise AssertionError('Current calibration must not emit or transport particles')


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden),patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        run()
