"""Update current-mode explanatory text after the scientific regression run.

The archived source and normalized AST comparison prove that only string
literals change. The actual offscreen menu and description are then read back.
No optical operator or numeric input is changed.
"""
import ast
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace


ROOT=Path(__file__).resolve().parents[3]
V=Path(__file__).resolve().parent/'validation'


def structure(text):
    tree=ast.parse(text)
    for item in ast.walk(tree):
        if isinstance(item,ast.Constant) and isinstance(item.value,str):
            item.value='<string>'
    return ast.dump(tree,include_attributes=False)


def strings(text):
    return [node.value for node in ast.walk(ast.parse(text))
        if isinstance(node,ast.Constant) and isinstance(node.value,str)]


def main():
    target=V/'ui-help-finalization.json'
    assert not target.exists(), 'Preserve the UI help check'
    history=V/'history/pre-helptext'
    history.mkdir(parents=True,exist_ok=False)
    replacements={
        'src/temsim/round_lens_models.py':(
            ('Calibrated material deflector and stigmator windings ',
             'Calibrated deflector, stigmator, quadrupole and hexapole windings '),
            ('unconfigured legacy sources and corrector multipoles retain their native models.',
             'unconfigured legacy sources retain their native models.')),
        'src/temsim/gui/simulation_menu.py':(
            ('Calibrated deflector and stigmator windings use their closed current circuits;',
             'Calibrated deflector, stigmator, quadrupole and hexapole windings use their closed current circuits;'),),
    }
    records=[]
    for name,edits in replacements.items():
        path=ROOT/name;raw=path.read_bytes();text=raw.decode('utf-8-sig')
        old_mtime=path.stat().st_mtime
        updated=text
        for old,new in edits:
            assert updated.count(old)==1
            updated=updated.replace(old,new)
        assert structure(text)==structure(updated), 'A non-string Python AST changed'
        old_literals,new_literals=strings(text),strings(updated)
        assert len(old_literals)==len(new_literals)
        diffs=[dict(before=a,after=b) for a,b in zip(old_literals,new_literals) if a!=b]
        assert len(diffs)==1
        saved=history/name;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(raw)
        path.write_bytes(updated.encode('utf-8'))
        records.append(dict(path=name,before_sha256=sha256(raw).hexdigest(),
            after_sha256=sha256(path.read_bytes()).hexdigest(),original_mtime=old_mtime,
            normalized_non_string_AST_equal=True,changed_string_literals=diffs))
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from temsim.gui.simulation_menu import SimulationMenu
    from temsim.round_lens_models import round_lens_model_description
    app=QApplication.instance() or QApplication([])
    state=SimpleNamespace(round_lens_field_model='copper_coil',gaussian_geometry_coupling='fixed_calibration')
    before=deepcopy(vars(state))
    description=round_lens_model_description(state)
    menu=SimulationMenu()
    tooltip=menu.round_lens_model_actions['copper_coil'].toolTip()
    assert all(term in value for term in ('quadrupole','hexapole') for value in (description,tooltip))
    assert 'corrector multipoles retain their native models' not in description
    assert vars(state)==before
    receipt=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-regression UI description strings only; no numeric/control AST changes',
        optical_state_unchanged=True,source_records=records,
        actual_description=description,actual_menu_tooltip=tooltip)
    target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('Verified current mode description/menu include calibrated quadrupoles and hexapoles; numerical AST unchanged')
    menu.deleteLater();app.processEvents()


if __name__=='__main__':
    main()
