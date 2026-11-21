"""Selectable material details with a link to the displayed source."""

from html import escape

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QLabel


class MaterialReferenceLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction
                                     | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.setOpenExternalLinks(True)

    def set_reference(self, description, source_url=""):
        lines = [escape(line) for line in description.splitlines()]
        if source_url:
            url = QUrl(str(source_url), QUrl.ParsingMode.StrictMode)
            if url.isValid() and url.scheme() in {"https", "http"} and url.host():
                href = escape(url.toString(QUrl.ComponentFormattingOption.FullyEncoded))
                title = "Open datasheet (PDF)" if url.path().lower().endswith(".pdf") else "Open source reference"
                lines.append(f'<a href="{href}" style="color: #7dd3fc">{title}</a>')
            else:
                lines.append(escape(str(source_url)))
        self.setText("<br>".join(lines))
