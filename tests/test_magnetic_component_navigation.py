"""Shared view controls and component navigation never start a ray calculation."""
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QDoubleSpinBox, QSizePolicy

from temsim.gui import main_window as shell
from temsim.gui.projection_angle_controls import ProjectionAngleControls


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "navigation.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    widget = shell.MainWindow()
    widget.preview_timer.stop()
    qtbot.addWidget(widget)

    def forbidden(*_args, **_kwargs):
        pytest.fail("A view or navigation action must not start a ray calculation")

    monkeypatch.setattr(widget.calculations.pool, "start", forbidden)
    return widget


def test_vector_rotation_joins_shared_projection_without_signal_echo(window, monkeypatch):
    workspace = window.workspace
    field = workspace.magnetic_field
    vectors = field.field_vectors
    received, requests = [], []
    original = vectors.set_projection_angle

    def receive(angle):
        received.append(angle)
        original(angle)

    monkeypatch.setattr(vectors, "set_projection_angle", receive)
    vectors.projection_angle_requested.connect(requests.append)
    workspace._set_projection_angle(27.5)
    assert received == [27.5]
    assert requests == []
    assert vectors.projection_controls.angle_deg == 27.5
    assert field.field_lines.canvas.projection_angle_deg == 27.5
    assert workspace.physical_layout.rotating_section._angle_deg == 27.5

    vectors.projection_angle_requested.emit(143.2)
    assert received == [27.5, 143.2]
    assert requests == [143.2]
    assert workspace._projection_angle_deg == 143.2
    assert field.field_lines.canvas.projection_angle_deg == 143.2
    assert workspace.physical_layout.rotating_section._angle_deg == 143.2

    # A section can still deliberately choose an independent observation angle.
    section = workspace.physical_layout.rotating_section
    section.follow_ray_diagram.setChecked(False)
    vectors.projection_angle_requested.emit(90.)
    assert section._angle_deg == 143.2
    assert field.field_lines.canvas.projection_angle_deg == 90.
    assert not window.preview_timer.isActive()


def test_longitudinal_angle_controls_sync_field_ray_and_physical_without_echo(window):
    workspace = window.workspace
    vectors = workspace.magnetic_field.field_vectors
    section = workspace.physical_layout.rotating_section
    field_requests, physical_requests = [], []
    vectors.projection_angle_requested.connect(field_requests.append)
    section.projection_angle_changed.connect(physical_requests.append)
    assert vectors.section.currentData() == "UZ"
    assert vectors.section.findData("XY") == -1
    assert section.follow_ray_diagram.isChecked()

    # The field control changes the common observation angle, not lens placement.
    vectors.projection_controls.slider.setValue(375)
    assert field_requests == [37.5]
    assert physical_requests == []
    assert workspace._projection_angle_deg == 37.5
    assert workspace.projection_slider.value() == 375
    assert vectors.canvas.projection_angle_deg == 37.5
    assert section.projection_controls.angle_deg == 37.5
    assert section._angle_deg == 37.5

    # Receiving a Physical Layout angle must not emit a field request in return.
    section.projection_controls.yz.click()
    assert field_requests == [37.5]
    assert physical_requests == [90.]
    assert workspace._projection_angle_deg == 90.
    assert workspace.projection_yz.isChecked()
    assert vectors.projection_controls.angle_deg == 90.
    assert vectors.canvas.projection_angle_deg == 90.
    assert vectors.section.currentData() == "UZ"

    # Ray Diagram remains a third equivalent entry point.
    workspace.projection_xz.click()
    assert field_requests == [37.5]
    assert physical_requests == [90.]
    assert workspace._projection_angle_deg == 0.
    assert vectors.projection_controls.angle_deg == 0.
    assert section.projection_controls.angle_deg == 0.

    # The 360° endpoint remains visible on every slider; the canvas may use 0°.
    vectors.projection_controls.slider.setValue(3600)
    assert field_requests == [37.5, 360.0]
    assert physical_requests == [90.]
    assert workspace._projection_angle_deg == 360.
    for control in (workspace.projection_controls, vectors.projection_controls,
                    section.projection_controls):
        assert control.angle_deg == 360.
        assert control.slider.value() == 3600
        assert control.xz.isChecked()
    assert vectors.canvas.projection_angle_deg == 0.
    workspace._set_projection_angle(37.25)
    for control in (workspace.projection_controls, vectors.projection_controls,
                    section.projection_controls):
        assert control.angle_deg == 37.25
        assert control.slider.value() == 373
    assert section._angle_deg == 37.25
    assert not window.preview_timer.isActive()


def test_shared_angle_controls_use_the_same_order_dimensions_and_tooltips(window):
    workspace = window.workspace
    controls = (workspace.projection_controls,
                workspace.magnetic_field.field_vectors.projection_controls,
                workspace.physical_layout.rotating_section.projection_controls)
    expected_tooltips = None
    expected_styles = None
    for control in controls:
        assert isinstance(control, ProjectionAngleControls)
        widgets = (control.label, control.xz, control.yz, control.slider)
        assert tuple(control.layout().itemAt(i).widget() for i in range(4)) == widgets
        assert control.layout().spacing() == 3
        assert control.layout().alignment() & Qt.AlignmentFlag.AlignLeft
        assert control.layout().contentsMargins().left() == 0
        assert control.label.text() == "Angle"
        assert [control.xz.text(), control.yz.text()] == ["X-Z", "Y-Z"]
        assert (control.slider.minimum(), control.slider.maximum()) == (0, 3600)
        assert (control.slider.minimumWidth(), control.slider.maximumWidth()) == (120, 180)
        assert (control.slider.singleStep(), control.slider.pageStep()) == (1, 50)
        assert not control.findChildren(QDoubleSpinBox)
        assert all(button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
                   for button in (control.xz, control.yz))
        styles = tuple(widget.styleSheet() for widget in (control, *widgets))
        if expected_styles is None:
            expected_styles = styles
        assert styles == expected_styles
        tooltips = tuple(widget.toolTip() for widget in widgets)
        if expected_tooltips is None:
            expected_tooltips = tooltips
        assert tooltips == expected_tooltips


def test_shared_angle_control_endpoint_and_receiving_values_do_not_echo(qtbot):
    control = ProjectionAngleControls()
    qtbot.addWidget(control)
    requests = []
    control.angle_changed.connect(requests.append)
    control.set_angle(37.5)
    control.set_angle(float("nan"))
    assert control.angle_deg == 37.5
    assert control.slider.value() == 375
    assert requests == []
    control.slider.setValue(3600)
    assert control.angle_deg == 360.0
    assert control.xz.isChecked()
    assert requests == [360.0]
    control.xz.click()
    assert control.angle_deg == 0.0
    assert control.slider.value() == 0
    assert control.xz.isChecked()
    assert requests == [360.0, 0.0]
    control.xz.click()
    assert control.xz.isChecked()
    assert requests == [360.0, 0.0]


def test_physical_menu_opens_vector_view_and_retains_real_component_key(window, monkeypatch):
    workspace = window.workspace
    layout, field = workspace.physical_layout, workspace.magnetic_field
    part = window.assembly.part("objective_lens")
    selected = []
    monkeypatch.setattr(field.field_vectors, "focus_component",
                        lambda key, z_mm=None: selected.append((key, z_mm)))
    workspace.tabs.setCurrentWidget(layout)
    menu = layout.component_navigation_menu(part.key, part.center_z_mm)
    destinations = {action.data(): action for action in menu.actions() if action.data()}
    assert {"ray", "parts", "vacuum", "magnetic"} <= destinations.keys()
    destinations["magnetic"].trigger()

    assert workspace.tabs.currentWidget() is field
    assert field.display_mode.currentData() == "vectors"
    assert field._selected_key == part.key
    assert selected[-1] == (part.key, part.center_z_mm)
    assert window._selected_component_key == part.key
    assert not window.preview_timer.isActive()
    menu.deleteLater()


def test_vector_navigation_reaches_physical_review_and_ray_diagram(window):
    workspace = window.workspace
    field = workspace.magnetic_field
    layout = workspace.physical_layout
    part = window.assembly.part("objective_lens")

    workspace.tabs.setCurrentWidget(field)
    layout.tabs.setCurrentWidget(layout.model_editor)
    # Use the menu's captured part identity, not a field-source label or region.
    field.field_vectors._state = window.state
    menu = field.field_vectors.component_navigation_menu(part.key)
    actions = {action.text(): action for action in menu.actions()}
    actions["Go to Physical Layout"].trigger()
    assert workspace.tabs.currentWidget() is layout
    assert layout.tabs.currentWidget() is layout.rotating_section
    assert layout.rotating_section._selected_key == part.key
    assert window._selected_component_key == part.key

    actions["Go to Ray Diagram"].trigger()
    assert workspace.tabs.currentWidget() is workspace.ray_page
    assert workspace._focused_part.key == part.key
    assert not window.preview_timer.isActive()
    menu.deleteLater()


def test_vector_selection_does_not_switch_pages_and_unknown_navigation_is_ignored(window):
    workspace = window.workspace
    field = workspace.magnetic_field
    workspace.tabs.setCurrentWidget(field)
    field.display_mode.setCurrentIndex(field.display_mode.findData("vectors"))

    # Sample ordinarily opens its dedicated page when selected from Ray Diagram.
    # A field/model selection must only synchronize the existing editor.
    field.field_vectors.component_selected.emit("sample")
    assert workspace.tabs.currentWidget() is field
    assert window._selected_component_key == "sample"
    field.field_vectors.navigation_requested.emit("missing_component", "physical", 123.)
    assert workspace.tabs.currentWidget() is field
    assert window._selected_component_key == "sample"
    assert not window.preview_timer.isActive()
    field.field_vectors._state = window.state
    field.field_vectors._fields_stale = True
    assert field.field_vectors.component_navigation_menu("sample") is None
