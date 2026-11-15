"""Render calibrated installed winding dialogs without source or field runs."""
from pathlib import Path
import os
from unittest.mock import patch

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from temsim.gui.transverse_winding_dialog import TransverseWindingConfigurationDialog
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.transverse_current import MATRIX_FIELD


def forbidden(*args,**kwargs):
    raise AssertionError('GUI rendering must not emit or transport a source')


def main():
    app=QApplication.instance() or QApplication([])
    if not QFontDatabase.families():
        font_path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/segoeui.ttf'
        font_id=QFontDatabase.addApplicationFont(str(font_path))
        assert font_id>=0,'A usable font is required for reviewable GUI output'
        app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0],9))
    document=read_document(INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml')
    parts={p['key']:p for p in document['parts']}
    target=Path(__file__).resolve().parent/'validation'
    target.mkdir(parents=True,exist_ok=True)
    for key,owner in (('probe_dph2_coils','probe_dph2_deflector'),
                      ('condenser_stigmator','condenser_stigmator')):
        part=parts[key]
        assert MATRIX_FIELD in part,'Installed source must be calibrated before recording its GUI'
        dialog=TransverseWindingConfigurationDialog(part,parts[owner],parts)
        dialog.resize(760,860)
        dialog.show()
        app.processEvents()
        assert dialog.apply_button.isEnabled()
        assert dialog.values()['turn_count_per_coil']==part['transverse_winding_turn_count_per_coil']
        assert all(edit.isEnabled() for edit in dialog.current_bias)
        destination=target/(key+'-winding-dialog.png')
        assert dialog.grab().save(str(destination))
        print(destination,flush=True)
        dialog.close()


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
