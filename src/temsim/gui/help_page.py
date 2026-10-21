"""Searchable, read-only in-app guide; independent of calculation state."""

from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QKeySequence, QShortcut, QTextDocument
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit,
    QListView, QListWidget, QListWidgetItem, QPushButton, QSplitter, QTextBrowser,
    QVBoxLayout, QWidget,
)

from temsim.gui.help_content import HELP_TOPICS
from temsim.branding import APP_NAME, APP_SUBTITLE


class HelpPage(QWidget):
    """Topic navigation and full-text search without loading a live instrument."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("helpPage")
        self._topics = {topic.key: topic for topic in HELP_TOPICS}
        self._search_text = {}
        self._items = {}

        title = QLabel(f"{APP_NAME} · Help")
        title.setObjectName("helpTitle")
        title.setStyleSheet("font-size: 21px; font-weight: 600; color: #f1f5f9;")
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        intro = QLabel(f"{APP_SUBTITLE}\nExplore the features, follow a workflow, or find a control by name.")
        intro.setWordWrap(True)
        intro.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        intro.setStyleSheet("color: #94a3b8;")

        self.search = QLineEdit()
        self.search.setObjectName("helpSearch")
        self.search.setAccessibleName("Search help")
        self.search.setPlaceholderText("Search features, controls or keywords…")
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip("Search all topic text. Ctrl+F focuses this field.")
        self.overview = QPushButton("Quick start")
        self.overview.setObjectName("helpQuickStart")
        self.overview.setAutoDefault(False)
        self.overview.clicked.connect(lambda: self.show_topic("quick_start"))
        search_row = QHBoxLayout()
        search_row.addWidget(self.search, 1)
        search_row.addWidget(self.overview)

        self.topics = QListWidget()
        self.topics.setObjectName("helpTopics")
        self.topics.setAccessibleName("Help topics")
        self.topics.setWordWrap(True)
        self.topics.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.topics.setResizeMode(QListView.ResizeMode.Adjust)
        self.topics.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.topics.setMinimumWidth(190)
        self.topics.setStyleSheet("""
            QListWidget { background: #0f172a; border: 1px solid #334155; }
            QListWidget::item { padding: 10px 8px; }
            QListWidget::item:selected { background: #1e3a5f; color: #f0f9ff; }
            QListWidget::item:hover:!selected { background: #1e293b; }
        """)
        self.browser = QTextBrowser()
        self.browser.setObjectName("helpBrowser")
        self.browser.setAccessibleName("Help article")
        self.browser.setMinimumWidth(300)
        self.browser.setReadOnly(True)
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.browser.setStyleSheet("""
            QTextBrowser {
                background: #0f172a; color: #e2e8f0;
                border: 1px solid #334155; padding: 12px;
                selection-background-color: #2563eb;
            }
        """)
        self.browser.document().setDefaultStyleSheet("""
            h1 { color: #f1f5f9; font-size: 22px; margin-bottom: 8px; }
            h2 { color: #bae6fd; font-size: 17px; margin-top: 22px; }
            h3 { color: #e2e8f0; font-size: 15px; margin-top: 18px; }
            p, li { line-height: 140%; }
            p.summary { color: #94a3b8; }
            a { color: #7dd3fc; }
            code { color: #c4b5fd; }
            th { background-color: #1e293b; }
        """)
        self.browser.anchorClicked.connect(self._open_link)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("helpTopicSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.topics)
        splitter.addWidget(self.browser)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 660])

        self.status = QLabel()
        self.status.setObjectName("helpSearchStatus")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #94a3b8;")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addLayout(search_row)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status)

        for topic in HELP_TOPICS:
            item = QListWidgetItem(topic.title)
            item.setData(Qt.ItemDataRole.UserRole, topic.key)
            item.setToolTip(topic.summary)
            self.topics.addItem(item)
            self._items[topic.key] = item
            document = QTextDocument()
            document.setHtml(topic.html)
            self._search_text[topic.key] = " ".join((
                topic.title, topic.summary, " ".join(topic.keywords),
                document.toPlainText(),
            )).casefold()
        self.topics.currentItemChanged.connect(self._topic_changed)
        self.search.textChanged.connect(self._filter_topics)
        self.find_shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        self.find_shortcut.activated.connect(self._focus_search)
        self._filter_topics("")

    def _focus_search(self) -> None:
        self.search.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.search.selectAll()

    def _filter_topics(self, query: str) -> None:
        terms = query.casefold().split()
        visible = []
        for key, item in self._items.items():
            matches = all(term in self._search_text[key] for term in terms)
            item.setHidden(not matches)
            if matches:
                visible.append(item)
        self.status.setText(
            f"{len(visible)} of {len(self._topics)} topics · Ctrl+F to search · Text can be selected and copied"
        )
        if not visible:
            self.topics.setCurrentRow(-1)
            self.browser.setHtml(
                "<h1>No matching topics</h1>"
                f"<p>No topic matches <b>{escape(query.strip())}</b>.</p>"
                "<p>Try a shorter phrase such as phase, aperture, scanning or corrector, "
                "or clear the search to see the full guide.</p>"
            )
        elif self.topics.currentItem() not in visible:
            self.topics.setCurrentItem(visible[0])

    def _topic_changed(self, item, _previous) -> None:
        if item is None:
            return
        topic = self._topics[item.data(Qt.ItemDataRole.UserRole)]
        self.browser.setHtml(
            f"<h1>{escape(topic.title)}</h1>"
            f'<p class="summary">{escape(topic.summary)}</p>'
            + topic.html
        )
        self.browser.verticalScrollBar().setValue(0)

    def show_topic(self, key: str) -> bool:
        item = self._items.get(key)
        if item is None:
            return False
        self.search.clear()
        self.topics.setCurrentItem(item)
        self.browser.verticalScrollBar().setValue(0)
        return True

    def _open_link(self, url: QUrl) -> None:
        if url.scheme() == "help":
            if self.show_topic(url.path().lstrip("/")) and url.fragment():
                self.browser.scrollToAnchor(url.fragment())
        elif not url.scheme() and not url.path() and url.fragment():
            self.browser.scrollToAnchor(url.fragment())


class HelpDialog(QDialog):
    """Reusable non-modal guide; closing it retains the current reading place."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("helpDialog")
        self.setWindowTitle(f"{APP_NAME} — Help")
        self.setModal(False)
        self.setMinimumSize(640, 440)
        available = self.screen().availableGeometry()
        self.resize(min(1000, max(640, available.width()-80)),
                    min(740, max(440, available.height()-80)))
        self.page = HelpPage(self)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setAutoDefault(False)
        buttons.rejected.connect(self.close)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.addWidget(self.page, 1)
        layout.addWidget(buttons)
