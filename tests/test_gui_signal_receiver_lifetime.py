"""Qt receiver ownership without optics calculations or a full MainWindow."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from shiboken6 import isValid

from temsim.gui.diagnostic_tabs import TransverseBeamView
from temsim.gui.electron_session_actions import ElectronSessionActions


class Emitter(QObject):
    pulse = Signal()


class SessionController(QObject):
    """Only the controller state used by the existing refresh action."""

    def __init__(self, parent):
        super().__init__(parent)
        self.refresh_reads = 0
        self.recalculations = 0
        self.selected_record = None
        self._shutdown = self._fields_stale = False
        self._scene = self._scene_request = None

    @property
    def records(self):
        self.refresh_reads += 1
        return ()

    def recalculate_historical(self):
        self.recalculations += 1


@pytest.fixture(params=("analysis", "hardware", "session"))
def receiver(request, qtbot):
    # Manage deletion here so pytest-qt does not close an already deleted
    # wrapper. Keep Python references alive throughout the ownership check.
    if request.param == "session":
        root = QWidget()
        owner = SessionController(root)
        helper = ElectronSessionActions(owner, QVBoxLayout(root))
    else:
        root = owner = TransverseBeamView()
        helper = getattr(root, request.param)
    yield SimpleNamespace(root=root, owner=owner, helper=helper, kind=request.param)
    if isValid(root):
        root.close()
        root.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_helpers_have_the_same_qt_lifetime_as_their_owner(receiver):
    helper = receiver.helper
    assert isinstance(helper, QObject)
    assert helper.parent() is receiver.owner
    if receiver.kind == "analysis":
        assert helper._resize_timer.parent() is helper


def test_owner_deletion_disconnects_two_surviving_external_senders(receiver):
    helper, owner = receiver.helper, receiver.owner
    senders = (Emitter(), Emitter())
    callback = helper.refresh if receiver.kind == "session" else helper.invalidate
    for sender in senders:
        sender.pulse.connect(callback)

    if receiver.kind == "session":
        before = owner.refresh_reads
        for sender in senders:
            sender.pulse.emit()
        assert owner.refresh_reads == before + 2
    else:
        for sender in senders:
            helper._cache_key = object()
            sender.pulse.emit()
            assert helper._cache_key is None

    owner.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(owner)
    # Python's reference is intentionally retained; Qt ownership must sever
    # delivery now, without waiting for cyclic GC or sender destruction.
    assert isinstance(helper, QObject) and not isValid(helper)
    before = owner.refresh_reads if receiver.kind == "session" else None
    marker = object()
    helper._cache_key = marker
    for sender in senders:
        assert isValid(sender)
        sender.pulse.emit()
    if receiver.kind == "session":
        assert owner.refresh_reads == before
    else:
        assert helper._cache_key is marker
