"""Destroy an idle workspace without retaining its detached scan controller."""
import gc
import weakref

import pyqtgraph as pg
from PySide6.QtCore import QCoreApplication, QEvent
from shiboken6 import isValid

from temsim.gui.visualization import VisualizationWorkspace


def _flush_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_detached_scan_panes_keep_controller_owned_and_release_all_views(qapp):
    """The empty scan shell must not become a surviving Qt signal source."""
    gc.collect()
    _flush_deletes()
    previous_views = {id(view) for view in pg.ViewBox.AllViews}

    def create_and_delete():
        workspace = VisualizationWorkspace()
        controller = workspace.scan_control
        refs = (
            weakref.ref(workspace),
            weakref.ref(controller),
            *(
                weakref.ref(view)
                for view in pg.ViewBox.AllViews
                if id(view) not in previous_views
            ),
        )
        try:
            assert controller.parent() is workspace
            assert controller.isHidden()
            assert controller.content_splitter.parent() is controller
            workspace.tabs.setCurrentWidget(workspace.scanning_page)
            # Moving the panes must preserve both their visibility and their
            # calculation route, even though the original shell is hidden.
            parameters = controller.parameters_page
            results = controller.result_tabs
            assert workspace.scanning_controls_tabs.widget(0) is parameters
            assert workspace.scanning_results_tabs is results
            assert parameters.isVisibleTo(workspace)
            assert results.isVisibleTo(workspace)
            requests = []
            workspace.calculation_requested.connect(requests.append)
            controller.calculation_requested.emit()
            assert requests == ["stem"]
        finally:
            workspace.close()
            workspace.deleteLater()
            _flush_deletes()
            # Retain Python wrappers here to check native ownership, rather
            # than depending on the later Python garbage collection.
            assert not isValid(workspace)
            assert not isValid(controller)
        return refs

    references = create_and_delete()
    gc.collect()
    _flush_deletes()
    gc.collect()
    assert all(reference() is None for reference in references)
    assert {id(view) for view in pg.ViewBox.AllViews} <= previous_views
