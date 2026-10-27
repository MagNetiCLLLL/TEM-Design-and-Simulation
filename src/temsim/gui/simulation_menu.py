"""Compact fidelity menu; model changes are explicit, not accuracy presets."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

from temsim.simulation_modes import (
    MODE_BY_KEY, FINITE_FIELD_KEYS, mode_key, linear_mode_issues, nonlinear_mode_issues,
)


class SimulationMenu(QMenu):
    mode_requested = Signal(str)
    configure_requested = Signal()
    particle_settings_requested = Signal()
    coherent_settings_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("Simulation", parent)
        self.setObjectName("simulationMenu")
        self.setToolTipsVisible(True)
        self.setStyleSheet("""
            QMenu::item { padding: 5px 22px 5px 24px; }
            QMenu::item:disabled { color: #94a3b8; }
            QMenu::item:selected:enabled { background: #2563eb; color: #ffffff; }
        """)
        self.addSection("Calculation model")
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_actions = {}
        def add_mode(key, menu, label=None):
            mode = MODE_BY_KEY[key]
            text = mode.label if mode.available else mode.label + " (not available)"
            action = QAction(label or text, self)
            action.setObjectName(f"simulationMode_{mode.key}")
            action.setCheckable(True)
            action.setData(mode.key)
            action.setEnabled(mode.available)
            action.setToolTip(mode.detail)
            action.triggered.connect(lambda checked, key=mode.key: self.mode_requested.emit(key) if checked else None)
            self.mode_group.addAction(action)
            menu.addAction(action)
            self.mode_actions[mode.key] = action
        add_mode("ideal_thin", self)
        self.finite_field_menu = self.addMenu("Finite-Field Optics")
        self.finite_field_menu.setObjectName("finiteFieldSourcesMenu")
        self.finite_field_menu.setToolTipsVisible(True)
        self.finite_field_menu.menuAction().setCheckable(True)
        self.finite_field_menu.addSection("Shared particle and wave execution")
        add_mode("unified", self.finite_field_menu, "Shared potentials and geometry")
        self.finite_field_menu.addSection("Legacy execution compatibility")
        add_mode("ideal", self.finite_field_menu)
        for key in ("linear_geometry", "nonlinear_material", "custom"):
            add_mode(key, self.finite_field_menu)
        add_mode("analytical", self.finite_field_menu)
        add_mode("coupled_multiphysics", self)
        self.addSeparator()
        configure = self.addAction("Configure lens models...")
        configure.triggered.connect(self.configure_requested.emit)
        self.addSection("Independent numerical settings")
        self.particle_settings_action = self.addAction("Particle: count and integration step...")
        self.particle_settings_action.setObjectName("particleNumericalSettingsAction")
        self.particle_settings_action.triggered.connect(self.particle_settings_requested.emit)
        self.coherent_settings_action = self.addAction("Coherent: grid, energy and memory...")
        self.coherent_settings_action.setObjectName("coherentNumericalSettingsAction")
        self.coherent_settings_action.triggered.connect(self.coherent_settings_requested.emit)
        note = self.addAction("High accuracy changes numerical resolution only")
        note.setEnabled(False)

    def set_state(self, state):
        current = mode_key(state)
        for key, action in self.mode_actions.items():
            action.setChecked(key == current)
        self.finite_field_menu.menuAction().setChecked(current in FINITE_FIELD_KEYS)
        # A retained linear shelf can be ready even when the current custom
        # configuration is different. This never runs the FEM solver.
        saved = getattr(state, "simulation_mode_profiles", {}).get("linear_geometry", {})
        descriptors = (state.lens_field_map_descriptors if current == "linear_geometry" else
                       saved.get("lens_field_map_descriptors", state.lens_field_map_descriptors))
        issues = linear_mode_issues(state, descriptors)
        action = self.mode_actions["linear_geometry"]
        action.setEnabled(not issues)
        action.setText("Linear Geometry Field (setup required)" if issues else "Linear Geometry Field")
        action.setToolTip(("Configure lens models first.\n" + "\n".join(issues[:5])) if issues else MODE_BY_KEY["linear_geometry"].detail)
        saved = getattr(state, "simulation_mode_profiles", {}).get("nonlinear_material", {})
        descriptors = (state.lens_field_map_descriptors if current == "nonlinear_material" else
                       saved.get("lens_field_map_descriptors", state.lens_field_map_descriptors))
        issues = nonlinear_mode_issues(state, descriptors)
        action = self.mode_actions["nonlinear_material"]
        action.setEnabled(not issues)
        action.setText("Nonlinear Material Field (setup required)" if issues else "Nonlinear Material Field")
        action.setToolTip(("Configure B-H fields first.\n" + "\n".join(issues[:5])) if issues else MODE_BY_KEY["nonlinear_material"].detail)
