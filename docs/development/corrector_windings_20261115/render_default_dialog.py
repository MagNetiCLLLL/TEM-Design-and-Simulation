"""Render the installed estimated N with unresolved metal area, without edits."""
from pathlib import Path
import os

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from temsim.gui.probe_winding_dialog import ProbeWindingConfigurationDialog
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT


app = QApplication.instance() or QApplication([])
# The Windows offscreen platform may not enumerate system fonts. Register the
# installed UI font explicitly for this receipt; production style is untouched.
if not QFontDatabase.families():
    font_path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/segoeui.ttf'
    font_id=QFontDatabase.addApplicationFont(str(font_path))
    assert font_id >= 0, 'A usable font is required for a reviewable UI receipt'
    app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9))
document = read_document(INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml')
part = next(p for p in document['parts'] if p['key']=='probe_hp2_hexapole')
dialog = ProbeWindingConfigurationDialog(part)
dialog.resize(680, 640)
dialog.show()
app.processEvents()
assert dialog.values()==dict(turn_count_per_coil=2, wire_cross_section_mm2=None)
destination = Path(__file__).resolve().parent/'validation/default-hp2-dialog.png'
assert dialog.grab().save(str(destination))
dialog.close()
print(destination)
