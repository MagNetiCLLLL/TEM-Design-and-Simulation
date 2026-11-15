"""Preserve original configs, optical controls and declared clipping geometry.

This one-time metadata capture does not emit particles, solve coil fields or
execute a coherent operator. Existing scientific receipts are not overwritten.
"""
from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from unittest.mock import patch

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.immutable_json import thaw_json
from temsim.module_manifest import read_document
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.posed_column_fields import capture_posed_column_fields
from temsim.physics.physical_scene import capture_physical_scene
from temsim.simulation_modes import switch_mode


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
BEFORE = REPORT / 'validation/before'
TARGET = re.compile(r'(probe|image)_(hp[12]|qpol|qpc|qph[12]|hpol|hpc)_(hexapole|quadrupole)$')


def forbidden(*args, **kwargs):
    raise AssertionError('Baseline metadata capture must not emit or transport source particles')


def normalized(value):
    return json.loads(json.dumps(value))


def geometry_content(state):
    # Retain declared clipping inputs without trying to admit Energy Filter
    # rectangular/slit openings to the circular wave-boundary executor.
    scene = capture_physical_scene(state)
    geometry = thaw_json(scene.payload['geometry'])
    geometry.pop('parts')
    geometry.pop('part_order')
    geometry['ownership'] = thaw_json(scene.payload['ownership'])
    return geometry


def main(*, resume_metadata=False):
    BEFORE.mkdir(parents=True, exist_ok=True)
    if resume_metadata:
        assert not (BEFORE/'optical-states.json').exists(), 'Do not overwrite preserved optical metadata'
        originals = json.loads((BEFORE/'inputs.json').read_text(encoding='utf-8'))
        inventory = json.loads((BEFORE/'all-configs.json').read_text(encoding='utf-8'))
        assert all(sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'] for row in inventory)
        capture_states(originals)
        return
    assert not (BEFORE / 'inputs.json').exists(), 'Do not overwrite a preserved baseline'
    originals, config_inventory = [], []
    for path in sorted((ROOT / 'configs').rglob('*.toml')):
        data = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        config_inventory.append(dict(path=relative, sha256=sha256(data).hexdigest()))
        if not relative.startswith('configs/instruments/column/'):
            continue
        document = read_document(path)
        keys = [row['key'] for row in document.get('parts', []) if TARGET.fullmatch(row['key'])]
        if not keys:
            continue
        destination = BEFORE / relative
        assert not destination.exists()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        originals.append(dict(path=relative, sha256=sha256(data).hexdigest(), hosts=keys))
    assert len(originals) == 5 and sum(len(row['hosts']) for row in originals) == 40
    (BEFORE / 'inputs.json').write_text(json.dumps(originals, indent=2)+'\n', encoding='utf-8')
    (BEFORE / 'all-configs.json').write_text(json.dumps(config_inventory, indent=2)+'\n', encoding='utf-8')
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    (BEFORE / 'working-tree.json').write_text(json.dumps(dict(head=git('rev-parse', 'HEAD'),
        branch=git('branch', '--show-current'),
        existing_pending_changes=git('status', '--short').splitlines(),
        policy='Preserve cumulative prior changes; latest task edits only named multipole configs and supporting code'),indent=2)+'\n',encoding='utf-8')
    capture_states(originals)


def capture_states(originals):
    catalog = AssemblyCatalog()
    states = []
    for column in catalog.columns:
        for recording in catalog.recording_systems:
            selection = AssemblySelection('FEG', column.name, recording.name)
            state = default_state()
            catalog.apply(state, selection)
            switch_mode(state, 'unified')
            if column.name == catalog.default_selection().column:
                apply_operating_mode_pair(state, 'nano_probe', 'diffraction',
                    column_name=column.name, recording_name=recording.name)
            apply_physical_layout_to_state(state)
            controls = normalized(state.to_dict())
            native_fields = capture_posed_column_fields(state, include_aligned=True)
            states.append(dict(selection=asdict(selection), optical_state=controls,
                native_fields=normalized([asdict(row) for row in native_fields]),
                clipping=normalized(geometry_content(state))))
            assert normalized(state.to_dict()) == controls
            print('Preserved controls, native fields and clipping:',column.name,'/',recording.name,flush=True)
    (BEFORE / 'optical-states.json').write_text(json.dumps(states,indent=2)+'\n',encoding='utf-8')
    print('Saved',len(originals),'affected sources, 40 multipole hosts,',len(states),'assembly states')


if __name__ == '__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--resume-metadata',action='store_true',help='Resume an incomplete metadata capture only if every config remains byte-identical')
    args=parser.parse_args()
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden):
        main(resume_metadata=args.resume_metadata)
