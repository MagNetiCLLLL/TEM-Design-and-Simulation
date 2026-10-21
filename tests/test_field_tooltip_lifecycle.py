"""A numerical tooltip may not survive the field snapshot it describes."""
import numpy as np
import pytest
from PySide6.QtCore import QEvent,QPointF,Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QToolTip

from temsim.gui.field_vector_canvas import FieldVectorCanvas
from temsim.gui.field_vector_page import FieldVectorPage
from test_field_density_canvas import section_sample


@pytest.mark.parametrize("style",("arrows","density"))
@pytest.mark.parametrize("action",("clear","replace","stale","invalidate"))
def test_visible_field_tooltip_is_hidden_when_its_snapshot_is_invalidated(qtbot,monkeypatch,style,action):
    page=FieldVectorPage()
    qtbot.addWidget(page)
    canvas=page.canvas
    canvas.resize(760,540)
    sample=section_sample("XZ")
    bounds=np.array([[-.001]*3,[.001]*3])
    canvas.set_sample(sample,bounds,"XZ")
    canvas.set_mode("B")
    canvas.set_render_style(style)
    canvas.grab()
    # Observe actual hover generation, but replace Qt's global tooltip surface
    # with a deterministic state so no native desktop tooltip is required.
    shown={"text":None,"hidden":0}
    monkeypatch.setattr(QToolTip,"showText",lambda _p,text,_w:shown.update(text=text))
    def hide():
        shown["text"]=None
        shown["hidden"]+=1
    monkeypatch.setattr(QToolTip,"hideText",hide)
    if style=="density":
        pos=canvas._plot_rect().center()
    else:
        index=int(canvas._visible_sample_indices[len(canvas._visible_sample_indices)//2])
        pos=QPointF(*canvas._screen_points[index])
    event=QMouseEvent(QEvent.Type.MouseMove,pos,pos,Qt.MouseButton.NoButton,
                     Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier)
    canvas.mouseMoveEvent(event)
    assert shown["text"] and "B" in shown["text"]
    before=shown["hidden"]
    if action=="clear":
        canvas.clear()
    elif action=="replace":
        canvas.set_sample(section_sample("XZ",strength=.5),bounds,"XZ")
    elif action=="stale":
        page.mark_inputs_stale()
    else:
        page.invalidate()
    assert shown["text"] is None and shown["hidden"]>before
    if action!="replace":
        assert canvas.sample is None and not len(canvas._visible_sample_indices)
    assert np.max(sample.b_t)==.25  # Old scientific arrays remain untouched.
