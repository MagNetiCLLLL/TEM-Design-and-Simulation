"""Explicit column-lens fidelity, independent of numerical accuracy and specimen physics.

The historical ``ideal`` key retains parameterised finite fields. The separate
``ideal_thin`` key selects the new thin-element particle approximation. Labels
never reinterpret saved mode keys. Mode shelves
store model settings and round-lens excitations, never resolved geometry.
"""

from copy import deepcopy
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SimulationMode:
    key: str
    label: str
    detail: str
    available: bool = True


MODES = (
    SimulationMode("ideal_thin", "Ideal Optics", "Fast thin-element particle optics. Lenses, stigmators and deflectors act at declared planes; field thickness and magnetic materials are not resolved. Positions and apertures remain active. Configured empirical Cs/Cc screens are inactive; saved values are retained. Coherent propagation requires Finite-Field Optics."),
    SimulationMode("unified", "Finite-Field Optics", "Particles and coherent waves use the same immutable electromagnetic potentials and finite geometry. Natural field aberrations replace empirical Cs/Cc screens and equivalent image lenses in the owned column interval; saved values remain unchanged. Unsupported source, field, geometry or wave configurations are rejected explicitly. This is a bounded forward-field model, not full instrument qualification."),
    SimulationMode("ideal", "Parameterized fields (legacy)", "Compatibility execution of parameterized finite column fields with configured aberrations, corrector multipoles and actual electron energies. Pole-piece shape and magnetic materials are not used to solve the field. The historical ideal key keeps its original behavior; it does not select the unified potential executor."),
    SimulationMode("analytical", "Analytical Field (compatibility)", "Compatibility alias for parameterized Finite-Field Optics. Existing profiles retain their key, configured aberrations and correctors. Saved geometry/map recipes are inactive."),
    SimulationMode("linear_geometry", "Linear Geometry Field", "Every enabled column round lens requires an explicit linear FEM recipe. Materials, mesh and geometry must be valid; no analytic fallback."),
    SimulationMode("nonlinear_material", "Nonlinear Material Field", "Static isotropic B-H fields, jointly solved at the complete current vector. Explicit reference materials required; no hysteresis or thermal coupling."),
    SimulationMode("coupled_multiphysics", "Coupled Multiphysics", "Future model: coupled electromagnetic, vacuum, thermal and mechanical behavior, including physically defined specimen drift. Not implemented; existing independent vacuum scattering is a separate approximation.", False),
    SimulationMode("custom", "Custom / Per-lens Models", "Each lens explicitly selects its analytic, generated or imported field model; not a uniform fidelity level."),
)
MODE_BY_KEY = {item.key: item for item in MODES}
FINITE_FIELD_KEYS = ("unified", "ideal", "linear_geometry", "nonlinear_material", "custom", "analytical")
MODEL_SETTINGS = ("lens_field_map_descriptors", "probe_aberrations", "image_aberrations",
                  "chromatic_aberration_enabled", "equivalent_image_lenses_enabled")


def validate_mode(value):
    key = str(value)
    if key not in MODE_BY_KEY:
        raise ValueError(f"Unknown simulation model: {key}")
    if not MODE_BY_KEY[key].available:
        raise ValueError(MODE_BY_KEY[key].detail)
    return key


def mode_key(state):
    return validate_mode(getattr(state, "simulation_mode", "custom"))


def is_ideal(state):
    """Historical finite-field flag; never reinterpret it as thin optics."""
    return mode_key(state) == "ideal"


def is_thin(state):
    return mode_key(state) == "ideal_thin"


def is_unified(state):
    """The shared potential executor, separate from archived compatibility modes."""
    return mode_key(state) == "unified"


def coherent_mode_issue(state):
    if is_thin(state):
        return ("Coherent propagation is unavailable in Ideal Optics (thin elements). "
                "Select Finite-Field Optics in Simulation; the mode was not changed automatically.")
    return None


def require_coherent_mode(state):
    issue = coherent_mode_issue(state)
    if issue:
        raise ValueError(issue)


def mode_display_label(state):
    key = mode_key(state)
    label = MODE_BY_KEY[key].label
    return f"Finite-Field Optics / {label}" if key in FINITE_FIELD_KEYS and key != "unified" else label


def uses_field_maps(state):
    return mode_key(state) in {"unified", "custom", "linear_geometry", "nonlinear_material"}


def normalise_profiles(profiles):
    if not isinstance(profiles, dict):
        raise ValueError("Simulation mode profiles must be a table")
    output = {}
    for key, row in profiles.items():
        validate_mode(key)
        if not isinstance(row, dict) or set(row) - {*MODEL_SETTINGS, "lens_excitation"}:
            raise ValueError("Simulation mode profile contains unsupported settings")
        row = deepcopy(row)
        for name in ("lens_field_map_descriptors", "probe_aberrations", "image_aberrations", "lens_excitation"):
            if name in row and not isinstance(row[name], dict):
                raise ValueError(f"Simulation profile {name} must be a table")
        for name in ("chromatic_aberration_enabled", "equivalent_image_lenses_enabled"):
            if name in row and not isinstance(row[name], bool):
                raise ValueError(f"Simulation profile {name} must be boolean")
        validate_model_recipes(row.get("lens_field_map_descriptors", {}))
        for values in row.get("lens_excitation", {}).values():
            if not isinstance(values, dict) or set(values) != {"percent", "polarity"}:
                raise ValueError("Lens excitation requires percent and polarity only")
            values["percent"] = float(values["percent"])
            if not math.isfinite(values["percent"]) or values["polarity"] not in {-1, 1}:
                raise ValueError("Lens excitation must be finite with polarity +1 or -1")
        output[key] = row
    return output


def validate_model_recipes(descriptors):
    """Check B-H snapshots at state/profile boundaries, without solving fields."""
    for row in descriptors.values():
        from temsim.excitation_calibration import validate_excitation_recipe
        validate_excitation_recipe(row)
        if isinstance(row, dict) and row.get("solver") == "axisymmetric_nonlinear_fem":
            from temsim.physics.nonlinear_circuits import operator_settings
            operator_settings(row)
            ni = float(row.get("ampere_turns", float("nan")))
            if not math.isfinite(ni) or ni < 0:
                raise ValueError("B-H ampere-turns at 100% must be finite and nonnegative")


def capture_mode_settings(state):
    settings = {name: deepcopy(getattr(state, name)) for name in MODEL_SETTINGS}
    settings["lens_excitation"] = {
        lens.key: {"percent": float(lens.percent), "polarity": int(lens.polarity)}
        for lens in state.lenses
    }
    return settings


def linear_mode_issues(state, descriptors=None):
    """Readiness check only: no field solve or invented material/current inputs."""
    descriptors = getattr(state, "lens_field_map_descriptors", {}) if descriptors is None else descriptors
    assembly = getattr(state, "_resolved_assembly", None)
    parts = {part.key: part for part in getattr(assembly, "parts", ())}
    issues = []
    for lens in state.lenses:
        if not bool(getattr(lens, "enabled", True)):
            continue
        row = descriptors.get(lens.key, {})
        if row.get("solver") != "axisymmetric_linear_fem":
            issues.append(f"{lens.key}: configure a linear geometry field")
        elif any(name not in row for name in ("relative_permeability", "ampere_turns")):
            issues.append(f"{lens.key}: material and ampere-turn inputs required")
        part = parts.get(lens.key)
        if part is None:
            issues.append(f"{lens.key}: resolved assembly geometry required")
        elif part.data.get("magnetic_circuit_topology") == "monolithic_saturated_insert":
            issues.append(f"{lens.key}: saturation insert cannot use a linear solver")
    return tuple(issues)


def switch_mode(state, target):
    """Switch atomically without solving, changing geometry or clearing result caches."""
    target, current = validate_mode(target), mode_key(state)
    if target == current:
        return False
    profiles = normalise_profiles(getattr(state, "simulation_mode_profiles", {}))
    outgoing = capture_mode_settings(state)
    incoming = deepcopy(profiles.get(target, outgoing))
    if target == "linear_geometry":
        issues = linear_mode_issues(state, incoming.get("lens_field_map_descriptors", {}))
        if issues:
            raise ValueError("Linear Geometry Field is not ready. " + "; ".join(issues[:4])
                             + (f"; {len(issues)-4} more issues" if len(issues) > 4 else ""))
    if target == "nonlinear_material":
        issues = nonlinear_mode_issues(state, incoming.get("lens_field_map_descriptors", {}))
        if issues:
            raise ValueError("Nonlinear Material Field is not ready. " + "; ".join(issues[:4]))
    profiles[current] = outgoing
    maps = getattr(state, "_simulation_mode_maps", {})
    maps[current] = dict(getattr(state, "_lens_field_map_bindings", {}))
    for name in MODEL_SETTINGS:
        if name in incoming:
            setattr(state, name, deepcopy(incoming[name]))
    # A model switch changes the field law, not the electrical operating point.
    # Historical shelves remain readable, but their stored currents are only
    # applied by an explicit operating-profile load.
    state.simulation_mode_profiles = profiles
    state.simulation_mode = target
    state._simulation_mode_maps = maps
    state._lens_field_map_bindings = dict(maps.get(target, maps[current]))
    # These adapters hold live objects; immutable unit-field/result caches remain.
    state._runtime_lens_field_provider_cache = {}
    return True


def nonlinear_mode_issues(state, descriptors=None):
    """Read-only setup validation, never a field solve."""
    from temsim.physics.nonlinear_circuits import operator_settings, SOLVER
    descriptors = state.lens_field_map_descriptors if descriptors is None else descriptors
    issues, operators = [], []
    for lens in state.lenses:
        row = descriptors.get(lens.key, {})
        if not lens.enabled and row.get("solver") != SOLVER:
            continue
        if row.get("solver") != SOLVER:
            issues.append(f"{lens.key}: configure a B-H geometry field")
            continue
        try:
            ni = float(row["ampere_turns"])
            if not math.isfinite(ni) or ni < 0:
                raise ValueError("nonnegative ampere-turns required")
            operators.append(operator_settings(row))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{lens.key}: {exc}")
    if operators and any(row != operators[0] for row in operators[1:]):
        issues.append("Joint B-H channels require identical material, mesh and solver settings")
    from temsim.lens_pose import lens_pose_registration
    poses = [lens_pose_registration(state, lens.key) for lens in state.lenses
             if descriptors.get(lens.key, {}).get("solver") == SOLVER]
    if poses and any(pose != poses[0] for pose in poses[1:]):
        issues.append("Joint B-H channels require the same rigid placement; independently moved lenses need "
                      "Analytical Field or a matching 3D field model")
    return tuple(issues)


def promote_custom_mode(state):
    """Explicit model editing keeps current controls, rather than restoring a shelf."""
    current = mode_key(state)
    if current == "unified":
        # Editing a retained potential recipe does not opt out of shared-field
        # execution. Capability validation takes place before propagation.
        state._runtime_lens_field_provider_cache = {}
        return
    if current != "custom":
        profiles = normalise_profiles(getattr(state, "simulation_mode_profiles", {}))
        profiles[current] = capture_mode_settings(state)
        state.simulation_mode_profiles = profiles
        state.simulation_mode = "custom"
        state._runtime_lens_field_provider_cache = {}
