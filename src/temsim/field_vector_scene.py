"""Selectable, captured production fields for electromagnetic diagnostics.

All positions are global right-handed XYZ metres, +Z downstream. B, E and
electron force use tesla, V/m and newtons respectively. Selection is a choice
of displayed contributions, never a change to the instrument or its solve.
The providers are the same providers used by particle propagation; this
module does not derive fields from the display mesh or solve a new B field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from temsim import input_io
from temsim.magnetic_field_scene import prepare_magnetic_scene
from temsim.physics.relativistic_lorentz import (
    ELECTRON, SPEED_OF_LIGHT_M_PER_S, momentum_from_kinetic_energy_ev,
    relativistic_gamma,
)


def _frozen(value, *, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _bounds(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (2, 3) or not np.isfinite(result).all() or np.any(result[1] <= result[0]):
        raise ValueError("Field bounds must be finite increasing XYZ metres")
    return _frozen(result)


def _inside(points, bounds):
    return np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)


def _union(bounds):
    return _bounds((np.min([b[0] for b in bounds], axis=0),
                    np.max([b[1] for b in bounds], axis=0)))


@dataclass(frozen=True)
class FieldVectorSource:
    """One selectable physical source or inseparable joint solution.

    ``component_keys`` lists logical channels belonging to this single entry.
    ``category`` is magnetic, electric, mixed or unmodelled. An unsupported
    entry is retained in the selection so it cannot silently become zero.
    """

    key: str
    label: str
    category: str
    component_keys: tuple[str, ...]
    bounds_m: np.ndarray
    model: str
    notes: tuple[str, ...] = ()
    supported: bool = True
    _magnetic: tuple = field(default=(), repr=False, compare=False)
    _electric: tuple = field(default=(), repr=False, compare=False)
    _unknown_b: bool = field(default=False, repr=False)
    _unknown_e: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class FieldVectorSample:
    points_m: np.ndarray
    b_t: np.ndarray
    e_v_per_m: np.ndarray
    force_n: np.ndarray
    magnetic_valid: np.ndarray
    electric_valid: np.ndarray
    force_valid: np.ndarray
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _ElectricSource:
    provider: object
    bounds_m: np.ndarray
    # An electric gun solution has a fixed cylindrical domain, and no inferred
    # outside zero. A Wien component uses its magnetic source's native support.
    magnetic_support: object | None = None

    def domain(self, points):
        if self.magnetic_support is not None:
            return self.magnetic_support.contains(points)
        valid = _inside(points, self.bounds_m)
        valid &= np.hypot(points[:, 0], points[:, 1]) <= self.bounds_m[1, 0]
        # Gun solvers intentionally allow constant-potential conductor samples
        # for interception. These are not a vacuum electron-force diagram.
        material = getattr(self.provider, "conductor_material_mask", None)
        if material is None:
            material = getattr(self.provider, "tip_material_mask", None)
        if callable(material) and valid.any():
            indices = np.flatnonzero(valid)
            mask = np.asarray(material(points[indices]), dtype=bool)
            if mask.shape != (len(indices),):
                raise ValueError("Electric material mask must have one value per point")
            valid[indices] &= ~mask
        return valid


def _magnetic_sample(sources, points, valid):
    """Apply the existing strict diagnostic active-source domain rule."""
    result = np.zeros_like(points)
    for source in sources:
        if source.known_zero:
            continue
        active = ((points[:, 2] >= source.bounds_m[0, 2])
                  & (points[:, 2] <= source.bounds_m[1, 2]))
        if source.field_map is None:
            active &= source.analytic_axial_mask(points)
        domain = source.contains(points)
        # A map's zero fill is an interpolation contract, not evidence of zero
        # field beyond its native domain. Never use a partial overlapping sum.
        valid &= ~active | domain
        evaluate = active & domain
        if evaluate.any():
            values = np.asarray(source.provider.field_at_global_positions_t(points[evaluate]), dtype=float)
            if values.shape != (int(evaluate.sum()), 3) or not np.isfinite(values).all():
                raise ValueError(f"{source.key}: magnetic provider returned invalid XYZ tesla values")
            result[evaluate] += values
    return result


def _electric_sample(sources, points, valid):
    result = np.zeros_like(points)
    for source in sources:
        domain = source.domain(points)
        if source.magnetic_support is None:
            valid &= domain
        else:
            support = source.magnetic_support
            active = ((points[:, 2] >= support.bounds_m[0, 2])
                      & (points[:, 2] <= support.bounds_m[1, 2]))
            if support.field_map is None:
                active &= support.analytic_axial_mask(points)
            valid &= ~active | domain
            domain &= active
        if domain.any():
            values = np.asarray(source.provider.field_at_global_positions_v_per_m(points[domain]), dtype=float)
            if values.shape != (int(domain.sum()), 3) or not np.isfinite(values).all():
                raise ValueError("Electric provider returned invalid XYZ V/m values")
            result[domain] += values
    return result


@dataclass(frozen=True)
class FieldVectorScene:
    sources: tuple[FieldVectorSource, ...]
    bounds_m: np.ndarray
    notes: tuple[str, ...]

    def sample(self, points_m, source_keys=None, *, kinetic_energy_ev=300000., direction=(0., 0., 1.)):
        """Sample the complete selected sum and a specified test-electron force.

        Duplicate keys are idempotent. Selecting a joint source always includes
        its entire solved field. Unselected sources contribute exactly zero by
        deliberate display choice; an unavailable selected source contributes
        NaN on its affected axial region. A field's finite axial support has
        the production zero continuation within the scene's display bounds.
        Off-axis/map/electric-domain failures remain unknown, never zero.

        The test electron has the specified *local* energy and direction at
        every query point. This does not reconstruct a transported trajectory
        or change the reference energy of effective magnetic providers.
        """
        points = np.asarray(points_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Field sample positions must be finite N by 3 XYZ metres")
        energy = float(kinetic_energy_ev)
        vector = np.asarray(direction, dtype=float)
        if not np.isfinite(energy) or energy < 0.:
            raise ValueError("Test electron kinetic energy must be finite and nonnegative eV")
        if vector.shape != (3,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("Test electron direction must be a finite nonzero XYZ vector")
        # Normalize before invoking the production momentum conversion to avoid
        # an overflowing norm for an otherwise finite user-supplied direction.
        vector = vector/np.max(np.abs(vector))
        vector = vector/np.linalg.norm(vector)
        available = {source.key: source for source in self.sources}
        keys = tuple(available) if source_keys is None else tuple(dict.fromkeys(source_keys))
        unknown = set(keys)-set(available)
        if unknown:
            raise ValueError(f"Unknown field source keys: {', '.join(sorted(map(str, unknown)))}")
        selected = tuple(available[key] for key in keys)
        b_valid = _inside(points, self.bounds_m)
        e_valid = b_valid.copy()
        for source in selected:
            affected = (points[:, 2] >= source.bounds_m[0, 2]) & (points[:, 2] <= source.bounds_m[1, 2])
            if source._unknown_b:
                b_valid &= ~affected
            if source._unknown_e:
                e_valid &= ~affected
        magnetic = tuple(item for source in selected for item in source._magnetic)
        electric = tuple(item for source in selected for item in source._electric)
        b = _magnetic_sample(magnetic, points, b_valid)
        e = _electric_sample(electric, points, e_valid)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            momentum = momentum_from_kinetic_energy_ev(energy, vector, ELECTRON)
            velocity = momentum/(float(relativistic_gamma(momentum, ELECTRON))*ELECTRON.rest_mass_kg)
        if not np.isfinite(velocity).all():
            raise ValueError("Test electron energy exceeds the finite numerical momentum range")
        force_valid = b_valid & e_valid
        force = ELECTRON.charge_c*(e + np.cross(velocity, b))
        b[~b_valid] = np.nan
        e[~e_valid] = np.nan
        force[~force_valid] = np.nan
        notes = tuple(dict.fromkeys((
            *self.notes, *(note for source in selected for note in source.notes),
            f"Selected {len(selected)} physical field source(s); omitted sources are deliberately excluded from this view.",
            f"Electron force q(E + v cross B): local kinetic energy {energy:g} eV, speed {np.linalg.norm(velocity)/SPEED_OF_LIGHT_M_PER_S:.8g} c; specified direction, not a traced orbit.",
        )))
        return FieldVectorSample(_frozen(points), _frozen(b), _frozen(e), _frozen(force),
                                 _frozen(b_valid, dtype=bool), _frozen(e_valid, dtype=bool),
                                 _frozen(force_valid, dtype=bool), notes)


def _component_bounds(component, fallback):
    """Mechanical label bounds only, never new field validity evidence."""
    center = float(getattr(component, "z_mm", getattr(component, "optical_reference_from_tip_mm", 0.)))
    length = float(getattr(component, "length_mm", getattr(component, "effective_length_mm", 1.)))
    if not np.isfinite((center, length)).all() or length <= 0.:
        return fallback
    bounds = np.array(fallback, copy=True)
    bounds[:, 2] = ((center-.5*length)*1e-3, (center+.5*length)*1e-3)
    return _bounds(bounds)


def _source_groups(state, magnetic):
    diagnostics = getattr(state, "_field_provider_diagnostics", {}) or {}
    groups = {}
    for source in magnetic._sources:
        provider = source.provider
        component = str(getattr(provider, "component_key", source.key.rsplit(":", 1)[0]))
        diagnostic = diagnostics.get(component, {})
        owner = str(diagnostic.get("joint_field_owner", component))
        groups.setdefault(owner, []).append(source)
    catalogue = []
    for key, sources in groups.items():
        channels = {key}
        joint = False
        for name, diagnostic in diagnostics.items():
            if str(diagnostic.get("joint_field_owner", "")) == key:
                joint = True
                channels.add(str(name))
                channels.update(map(str, diagnostic.get("channels", ())))
        for source in sources:
            channels.update(map(str, getattr(source.provider, "drive_keys", ())))
        notes = []
        if joint:
            notes.append("Inseparable nonlinear joint magnetic circuit: all listed channels share one solved field, counted once; individual saturated fields cannot be added.")
        elif len(channels) > 1:
            notes.append("Logical drive channels share the same physical coils; their production drive is already summed and is counted once.")
        identities = tuple(getattr(source, "identity", None) for source in sources)
        models = tuple(dict.fromkeys(getattr(identity, "model", "unknown") for identity in identities))
        notes.extend(getattr(identity, "limitation", "Unknown provider model identity") for identity in identities)
        for source in sources:
            notes.append(f"{source.key}: native valid radial limit {source.radius_m*1e3:.6g} mm; global axial support {source.bounds_m[0, 2]*1e3:.6g} to {source.bounds_m[1, 2]*1e3:.6g} mm. Registered map/pose boundaries are checked point by point.")
        label = sources[0].label
        if joint:
            label += " — joint circuit: " + ", ".join(sorted(channels))
        elif len(channels) > 1:
            label += " — shared: " + ", ".join(sorted(channels))
        electric = tuple(_ElectricSource(source.provider, source.bounds_m, source)
                         for source in sources if source.category == "crossed_field"
                         and callable(getattr(source.provider, "field_at_global_positions_v_per_m", None)))
        catalogue.append(FieldVectorSource(key, label, "mixed" if electric else "magnetic",
                                          tuple(sorted(channels)), _union([source.bounds_m for source in sources]),
                                          "joint_nonlinear" if joint else " + ".join(models),
                                          tuple(dict.fromkeys(notes)), True, tuple(sources), electric))
    return catalogue


@input_io.using_state_inputs
def prepare_field_vector_scene(state, *, magnetic_scene=None, z_limits_mm=None):
    """Capture selectable production B/E providers in a numerical worker.

    This can reuse/create the canonical gun electric solution, so callers must
    not execute it on the GUI thread. Magnetic FEM recipes are never solved by
    this view: missing/stale captures remain explicitly unavailable. Plot
    limits do not change electric solve boundaries or source excitation.
    """
    limits = None
    if z_limits_mm is not None:
        limits = np.asarray(z_limits_mm, dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] <= limits[0]:
            raise ValueError("Field vector limits must be two increasing finite millimetre positions")
        limits = limits*1e-3
    notes = ["Captured production providers; global right-handed XYZ metres, +Z downstream. B in T, E in V/m; these different units are never added directly.",
             "Unknown field regions are NaN. Analytic near-axis bounds and registered map domains are model limits, not measured accuracy or a coil-resolved FEM claim.",
             "Passive apertures, supports and insulation are not independent field sources; they enter only through existing solver boundary/material models."]
    magnetic_error = None
    from temsim.optics.model import State
    # A displayed magnetic profile may have clipped out entire components, or
    # belong to an earlier input snapshot. Recapture the complete current graph
    # from the same input owner as E. This only reads dependency-checked B caches
    # and cheap analytic providers; it cannot start a magnetic FEM solve.
    if magnetic_scene is not None and isinstance(state, State):
        try:
            complete = prepare_magnetic_scene(state)
        except (ValueError, RuntimeError, NotImplementedError, OSError) as error:
            magnetic_scene = None
            magnetic_error = str(error)
        else:
            if (magnetic_scene.numerical_identity is None
                    or magnetic_scene.numerical_identity != complete.numerical_identity):
                notes.append("The supplied magnetic view was cropped, stale or unverified; the complete graph was recaptured from the same instrument inputs as E before sampling.")
                magnetic_scene = complete
    if magnetic_scene is None and magnetic_error is None:
        try:
            magnetic_scene = prepare_magnetic_scene(state)
        except (ValueError, RuntimeError, NotImplementedError, OSError) as error:
            magnetic_error = str(error)
    catalogue = _source_groups(state, magnetic_scene) if magnetic_scene is not None else []
    if magnetic_scene is not None:
        notes.extend(magnetic_scene.notes)
        fallback = magnetic_scene.bounds_m
    else:
        fallback = _bounds(((-1e-3, -1e-3, 0.), (1e-3, 1e-3, 3.)))

    electric_capture = None
    if getattr(state, "electron_gun", None) is not None:
        try:
            from temsim.physics.instrument_electric import capture_instrument_electric_field
            electric_capture = capture_instrument_electric_field(state)
        except (ValueError, RuntimeError, NotImplementedError, OSError) as error:
            note = f"Instrument electrostatic capture unavailable: {error}. No zero electric field is inferred."
            catalogue.append(FieldVectorSource("gun_electric_field", "Gun electrostatics — unavailable", "unmodelled",
                                               ("gun_electric_field",), _bounds(fallback), "unavailable_electric", (note,),
                                               False, _unknown_e=True))
        else:
            bounds = _bounds(electric_capture.bounds_m)
            gun = electric_capture.gun_snapshot
            channels = {"gun_electric_field"}
            for name in ("extractor", "electrostatic_lens", "accelerator"):
                part = getattr(gun, name, None)
                if part is not None:
                    channels.add(str(part.key))
            if getattr(gun, "emitter", None) is not None:
                channels.add(str(getattr(gun.emitter, "key", "emitter")))
            base = electric_capture.base_field
            # Boundary rings, including a common-potential Wien housing, are
            # participants in this joint Laplace solution. They are distinct
            # from the independently driven transverse Wien contribution.
            for row in getattr(base, "request", {}).get("rings", ()):
                channels.add(str(row["key"] if isinstance(row, dict) else row[0]))
            model = type(base).__name__
            source_notes = (*electric_capture.notes,
                            "One joint electrostatic boundary solution; electrode contributions are not independently isolated. Gun base excludes the separately selectable Wien field.",
                            "The electric solution has a fixed cylindrical domain; outside support and conductor interiors are not admitted as vacuum electron-force samples.")
            catalogue.append(FieldVectorSource("gun_electric_field", "Gun electrodes / grounded column — joint E", "electric",
                                               tuple(sorted(channels)), bounds, model, source_notes, True,
                                               _electric=(_ElectricSource(base, bounds),)))
            wien = getattr(electric_capture.provider, "wien_field", None)
            if wien is not None and not any(entry.category == "mixed" for entry in catalogue):
                # A failed unrelated lens capture must not drop the Wien part
                # from an otherwise successful production electric capture.
                from temsim.magnetic_field_scene import _Source, _component_radius_m
                component = getattr(getattr(gun, "monochromator", None), "wien", None)
                if component is not None:
                    low, high = np.asarray(component.field_support_mm, dtype=float)*1e-3
                    radius = _component_radius_m(component, axial_scale_mm=component.active_length_mm)
                    support = _bounds(((-radius, -radius, low), (radius, radius, high)))
                    captured_wien = _Source(str(component.key), wien, support, radius,
                                            category="crossed_field", label=str(component.name))
                    catalogue.append(FieldVectorSource(str(component.key), str(component.name), "mixed", (str(component.key),),
                                                       support, type(wien).__name__,
                                                       ("Existing captured Wien E/B provider; finite native support and near-axis radius retained.",),
                                                       True, (captured_wien,), (_ElectricSource(wien, support, captured_wien),)))
                else:
                    catalogue.append(FieldVectorSource("unavailable_wien_field", "Wien field — support unavailable", "unmodelled", (),
                                                       bounds, "unavailable_electric", ("Captured electric solution includes Wien, but no explicit component support is available.",),
                                                       False, _unknown_e=True))

    all_bounds = [entry.bounds_m for entry in catalogue] or [fallback]
    bounds = _union(all_bounds)
    if limits is not None:
        view = np.array(bounds, copy=True)
        view[:, 2] = limits
        bounds = _bounds(view)
    if magnetic_error is not None:
        catalogue.insert(0, FieldVectorSource("unavailable_magnetic_field", "Magnetic field — recapture required", "unmodelled",
                                              (), bounds, "unavailable_magnetic", (magnetic_error,), False, _unknown_b=True))

    # Missing E capture must stay unknown across the entire requested view,
    # including locations beyond the available magnetic graph's support.
    catalogue = [FieldVectorSource(entry.key, entry.label, entry.category, entry.component_keys,
                                    bounds if entry.model == "unavailable_electric" else entry.bounds_m,
                                    entry.model, entry.notes, entry.supported, entry._magnetic, entry._electric,
                                    entry._unknown_b, entry._unknown_e) for entry in catalogue]

    # A magnet's grounded housing may be an electrostatic boundary participant;
    # that does not establish that its independent B contribution was captured.
    represented = {key for entry in catalogue if entry._magnetic or entry._unknown_b
                   for key in entry.component_keys}
    components = [*getattr(state, "lenses", ()), *getattr(state, "stigmators", ()),
                  *getattr(state, "corrector_elements", ()), *getattr(state, "deflectors", ())]
    gun = getattr(state, "electron_gun", None)
    components.extend(component for component in (getattr(gun, "deflector", None), getattr(gun, "stigmator", None))
                      if component is not None)
    for component in components:
        key = str(component.key)
        if key in represented:
            continue
        represented.add(key)
        label = str(getattr(component, "name", key))
        # DP12 is explicitly a layout-only reference with no interaction. Do
        # not turn its enabled display switch into a missing physical magnet.
        interaction = getattr(component, "interaction_kind", getattr(component, "INTERACTION_KIND", None))
        if interaction == "none":
            catalogue.append(FieldVectorSource(key, label+" — reference only", "magnetic", (key,),
                                               _component_bounds(component, bounds), "reference_zero",
                                               ("Declared non-interacting layout reference: no independent field contribution.",)))
        elif not bool(getattr(component, "enabled", True)):
            catalogue.append(FieldVectorSource(key, label+" — disabled", "magnetic", (key,),
                                               _component_bounds(component, bounds), "disabled_zero",
                                               ("Disabled production component: exact zero contribution; no active field geometry is inferred.",)))
        elif magnetic_error is None:
            # Mechanical length does not delimit an unmodelled fringe field.
            # Its missing contribution invalidates the selected total across
            # the view until an actual provider supplies a supported domain.
            catalogue.append(FieldVectorSource(key, label+" — magnetic provider unavailable", "unmodelled", (key,),
                                               bounds, "unavailable_magnetic",
                                               ("Enabled component is absent from the captured magnetic graph; no independent provider or finite field extent is declared. Its selected magnetic contribution remains unknown.",),
                                               False, _unknown_b=True))

    unsupported = []
    pulser = getattr(state, "nanopulser", None)
    if pulser is not None and bool(getattr(pulser, "installed", False)) and bool(getattr(pulser, "blanked", False)):
        entrance = (float(pulser.z_mm)-.5*float(pulser.plate_length_mm))*1e-3
        unsupported.append(("electrostatic_blanker", "Electrostatic blanker", entrance, False, True,
                            "Active blanker has only a thin-kick model, no continuous E provider; E and force are unknown from its entrance downstream."))
    if bool(getattr(state, "energy_filter_installed", False)):
        entrance = next((float(part.z_mm)*1e-3 for part in getattr(state, "apertures", ())
                         if str(part.key) == "energy_filter_entrance_aperture" and bool(getattr(part, "installed", False))), None)
        unsupported.append(("energy_filter", "Bent energy filter", bounds[0, 2] if entrance is None else entrance,
                            True, True, "Bent post-column coordinates have no straight-column vector provider; fields are unknown from the entrance downstream."))
    for key, label, entrance, unknown_b, unknown_e, note in unsupported:
        if not np.isfinite(entrance):
            raise ValueError(f"{key}: unsupported field entrance must be finite")
        affected = np.array(bounds, copy=True)
        affected[0, 2] = entrance
        affected[1, 2] = max(bounds[1, 2], entrance+1e-6)
        catalogue.append(FieldVectorSource(key, label+" — unmodelled", "unmodelled", (key,), _bounds(affected),
                                           "unmodelled", (note,), False, _unknown_b=unknown_b, _unknown_e=unknown_e))
    return FieldVectorScene(tuple(catalogue), bounds, tuple(dict.fromkeys(notes)))


__all__ = ["FieldVectorSource", "FieldVectorSample", "FieldVectorScene", "prepare_field_vector_scene"]
