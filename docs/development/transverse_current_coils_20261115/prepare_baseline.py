"""Preserve pre-migration source files and authoritative native optical states."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from unittest.mock import patch

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.posed_column_fields import capture_posed_column_fields
from temsim.simulation_modes import switch_mode
from temsim.transverse_winding import supported_transverse_winding


ROOT=Path(__file__).resolve().parents[3]
REPORT=Path(__file__).resolve().parent
BEFORE=REPORT/'validation/before'


def forbidden(*args,**kwargs):
    raise AssertionError('Field/CAD setup must not emit or transport source particles')


def normalized(value):
    return json.loads(json.dumps(value))


def main():
    assert not (BEFORE/'optical-states.json').exists(), 'Preserve the original baseline'
    records=[]
    for path in sorted((ROOT/'configs').rglob('*.toml')):
        document=tomllib.loads(path.read_text(encoding='utf-8'))
        keys=[p['key'] for p in document.get('parts',[]) if supported_transverse_winding(p)]
        if not keys: continue
        relative=path.relative_to(ROOT)
        target=BEFORE/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists(), 'Do not overwrite a preserved input'
        target.write_bytes(path.read_bytes())
        records.append(dict(path=relative.as_posix(),sha256=sha256(path.read_bytes()).hexdigest(),hosts=keys))
    catalog=AssemblyCatalog();states=[]
    for column in catalog.columns:
        for recording in catalog.recording_systems:
            state=default_state()
            selection=AssemblySelection('FEG',column.name,recording.name)
            catalog.apply(state,selection);switch_mode(state,'unified')
            if column.name==catalog.default_selection().column:
                apply_operating_mode_pair(state,'nano_probe','diffraction',
                    column_name=column.name,recording_name=recording.name)
            apply_physical_layout_to_state(state)
            original=normalized(state.to_dict())
            fields=capture_posed_column_fields(state,include_aligned=True)
            states.append(dict(selection=asdict(selection),optical_state=original,
                native_fields=normalized([asdict(f) for f in fields])))
            assert original==normalized(state.to_dict())
            print('Preserved native inputs:',column.name,'/',recording.name,flush=True)
    BEFORE.mkdir(parents=True,exist_ok=True)
    (BEFORE/'inputs.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    (BEFORE/'optical-states.json').write_text(json.dumps(states,indent=2)+'\n',encoding='utf-8')
    print(f'Preserved {len(records)} source files and {len(states)} optical assembly states')


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
