"""Record detached default-command currents without transport or field solving."""
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.transverse_coil_field import capture_transverse_coil_fields
from temsim.simulation_modes import switch_mode


def forbidden(*args, **kwargs):
    raise AssertionError('Current readback must not emit or transport particles')


def main():
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    switch_mode(state, 'unified')
    apply_operating_mode_pair(state, 'nano_probe', 'diffraction',
        column_name=selection.column, recording_name=selection.recording)
    apply_physical_layout_to_state(state)
    state.round_lens_field_model = 'copper_coil'
    before = json.dumps(state.to_dict(), sort_keys=True)
    fields = capture_transverse_coil_fields(state, time_s=0., include_gun=True)
    rows = [dict(source_id=f.potential.source_id, component_key=f.component_key,
        material_host=f.potential.provenance['material_host'],
        turns_per_coil=f.potential.turn_count_per_coil,
        channel_currents_a=list(f.potential.currents_a),
        dynamic=f.dynamic, drive_keys=list(f.drive_keys),
        captured_time_s=f.captured_time_s) for f in fields]
    assert json.dumps(state.to_dict(), sort_keys=True) == before
    receipt = dict(status='passed', selection=asdict(selection),
        operating_modes=['nano_probe', 'diffraction'], field_model='copper_coil',
        time_s=0., scope='Detached current readback only; no particle, field-grid or coherent calculation',
        optical_state_unchanged=True, sources=rows)
    path = Path(__file__).resolve().parent / 'validation/default-command-currents.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(f'Recorded {len(rows)} station sources; {sum(any(r["channel_currents_a"]) for r in rows)} powered')


if __name__ == '__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden):
        main()
