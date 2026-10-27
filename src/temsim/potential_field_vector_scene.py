"""Display the exact retained E/B sources consumed by potential executors.

The explicit XYZ display box is not a field-validity domain. Unavailable
queries remain NaN, including an unknown FEM exterior. No field solve,
independent B interpolation or inference from a CAD mesh occurs here.
"""
from dataclasses import dataclass

import numpy as np

from temsim.field_vector_scene import FieldVectorScene, FieldVectorSource, FieldVectorSample, _bounds, _frozen, _inside
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
from temsim.physics.relativistic_lorentz import (
    ELECTRON, SPEED_OF_LIGHT_M_PER_S, momentum_from_kinetic_energy_ev, relativistic_gamma,
)


@dataclass(frozen=True)
class _PotentialFieldVectorSource(FieldVectorSource):
    # Display-only sampling hints are distinct from field validity. In
    # particular a Gaussian's useful native window is not a support boundary.
    sampling_bounds_m: tuple = ()
    support_planes_m: tuple = ()


def _cylinder_hints(registration, radius, axial, *, finite_faces=True):
    """Enclose one posed local cylinder; retain its actual axial planes."""
    rotation, origin = registration.rotation_array, registration.origin_array_m
    normal = rotation[:, 2]
    lower, upper = np.asarray(axial, dtype=float)
    centre = origin+normal*(lower+upper)*.5
    extent = radius*np.hypot(rotation[:, 0], rotation[:, 1])+abs(normal)*(upper-lower)*.5
    bounds = tuple(tuple(float(value) for value in row) for row in (centre-extent, centre+extent))
    planes = (tuple((*normal, float(z+np.dot(normal, origin)))) for z in (lower, upper))
    return bounds, tuple(planes) if finite_faces else ()


def _sampling_hints(potential):
    """Capture known source extents without sampling, extrapolating or solving.

    Bounds only place useful grid anchors. Planes separately prohibit texture
    interpolation across compact activation faces and finite domain ends.
    Actual point validity always comes from the retained potential itself.
    """
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.gun_magnetic_potential import GunMagneticPotential
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
    from temsim.physics.regular_axisymmetric_magnetic_potential import RegularAxisymmetricMagneticPotential

    kind = type(potential)
    if kind in {AnalyticMagneticPotential, AnalyticPotentialDomain, HighOrderRoundLensPotential}:
        native = (potential.base.field if kind is AnalyticPotentialDomain else
                  potential.native if kind is HighOrderRoundLensPotential else potential.field)
        all_z = ((kind is AnalyticPotentialDomain and type(native) is FrozenAnalyticField)
                 or (kind is HighOrderRoundLensPotential and potential.axial_domain == "gaussian_all_z"))
        radius = potential.valid_radius_m if kind is HighOrderRoundLensPotential else native.radial_support_m
        bounds, planes = _cylinder_hints(native.registration, radius,
            np.asarray(native.native_support_mm)*.001, finite_faces=not all_z)
        return (bounds,), planes
    if kind in {AxisymmetricMagneticPotential, RegularAxisymmetricMagneticPotential}:
        original = potential.source if kind is RegularAxisymmetricMagneticPotential else potential
        radius = potential.valid_radius_m if kind is RegularAxisymmetricMagneticPotential else original.r_m[-1]
        bounds, planes = _cylinder_hints(original.registration, radius, original.z_m[[0, -1]])
        return (bounds,), planes
    if kind is ElectricPotentialSnapshot:
        domain = potential.metadata["domain"]
        bounds, _ = _cylinder_hints(CoordinateRegistration(), domain["radius_m"], domain["mesh_z_interval_m"])
        # A declared constant-potential continuation may start at a mesh end
        # inside the full domain. Keep that field transition as well.
        ends = (*domain["mesh_z_interval_m"], domain["z_min_m"], domain["z_max_m"])
        planes = tuple((0., 0., 1., float(z)) for z in sorted({z for z in ends if z is not None}))
        return (bounds,), planes
    if kind is GunMagneticPotential:
        bounds, planes = [], []
        for centre, half, edge, *_ in potential.terms:
            box, faces = _cylinder_hints(CoordinateRegistration(), potential.radial_support_m,
                                        (centre-half-edge, centre+half+edge))
            bounds.append(box)
            planes.extend(faces)
        return tuple(bounds), tuple(planes)
    raise ValueError("Retained potential requires an explicit display sampling-hint adapter")


def _source_samples(snapshot, source_id, points, name):
    """Use vector batches when admitted; isolate unavailable display points.

    The authoritative potential API deliberately rejects an entire mixed-validity
    batch. Bisection retains that strict API while exposing per-point display
    validity. A failed point is never a zero source or a partial field sum.
    """
    values = np.zeros_like(points)
    valid = np.zeros(len(points), dtype=bool)
    errors = set()

    def visit(indices):
        if not len(indices):
            return
        try:
            sample = np.asarray(snapshot.sample_source(source_id, points[indices])[name])
            if sample.shape != (len(indices), 3) or not np.isfinite(sample).all():
                raise ValueError("Retained potential returned invalid vector samples")
        except ValueError as error:
            if len(indices) == 1:
                errors.add(str(error))
            else:
                middle = len(indices)//2
                visit(indices[:middle])
                visit(indices[middle:])
        else:
            values[indices], valid[indices] = sample, True

    visit(np.arange(len(points)))
    return values, valid, tuple(sorted(errors))


@dataclass(frozen=True)
class PotentialFieldVectorScene(FieldVectorScene):
    snapshot: ElectromagneticPotentialSnapshot

    def sample(self, points_m, source_keys=None, *, kinetic_energy_ev=300000., direction=(0., 0., 1.)):
        points = np.asarray(points_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Field sample positions must be finite N by 3 XYZ metres")
        if isinstance(kinetic_energy_ev, (bool, np.bool_)):
            raise ValueError("Test electron energy must be a finite nonnegative real value")
        energy, vector = float(kinetic_energy_ev), np.asarray(direction, dtype=float)
        if not np.isfinite(energy) or energy < 0:
            raise ValueError("Test electron energy must be a finite nonnegative real value")
        if vector.shape != (3,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("Test electron direction must be a finite nonzero XYZ vector")
        vector = vector/np.max(abs(vector))
        vector /= np.linalg.norm(vector)
        available = {source.key: source for source in self.sources}
        keys = tuple(available) if source_keys is None else tuple(dict.fromkeys(source_keys))
        if set(keys)-set(available):
            raise ValueError("Unknown retained-potential field source")
        by_key = {row.source_id: row for row in self.snapshot.contributions}
        b, electric = np.zeros_like(points), np.zeros_like(points)
        b_valid = _inside(points, self.bounds_m)
        e_valid = b_valid.copy()
        notes = list(self.notes)
        for key in keys:
            row = by_key[key]
            name = "B_t" if row.kind == "magnetic" else "E_v_per_m"
            indices = np.flatnonzero(_inside(points, self.bounds_m))
            values, admitted, errors = _source_samples(self.snapshot, key, points[indices], name)
            if row.kind == "magnetic":
                b[indices] += values
                b_valid[indices] &= admitted
            else:
                electric[indices] += values
                e_valid[indices] &= admitted
            notes.extend(f"{key}: unavailable retained-potential samples: {error}" for error in errors)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            momentum = momentum_from_kinetic_energy_ev(energy, vector, ELECTRON)
            velocity = momentum/(float(relativistic_gamma(momentum, ELECTRON))*ELECTRON.rest_mass_kg)
        if not np.isfinite(velocity).all():
            raise ValueError("Test electron energy exceeds the finite numerical momentum range")
        b_valid &= np.isfinite(b).all(axis=1)
        e_valid &= np.isfinite(electric).all(axis=1)
        force_valid = b_valid & e_valid
        with np.errstate(over="ignore", invalid="ignore"):
            force = ELECTRON.charge_c*electric+np.cross(velocity, ELECTRON.charge_c*b)
        force_valid &= np.isfinite(force).all(axis=1)
        b[~b_valid], electric[~e_valid], force[~force_valid] = np.nan, np.nan, np.nan
        notes.extend((f"Selected {len(keys)} retained physical source(s); each joint source is counted once.",
                      f"Local test-electron force q(E + v cross B), {energy:g} eV, speed {np.linalg.norm(velocity)/SPEED_OF_LIGHT_M_PER_S:.8g} c; not a traced orbit."))
        return FieldVectorSample(_frozen(points), _frozen(b), _frozen(electric), _frozen(force),
            _frozen(b_valid, dtype=bool), _frozen(e_valid, dtype=bool), _frozen(force_valid, dtype=bool), tuple(dict.fromkeys(notes)))


def prepare_potential_field_vector_scene(snapshot, display_bounds_m):
    """Prepare a selectable view without recapturing or solving the instrument."""
    if type(snapshot) is not ElectromagneticPotentialSnapshot:
        raise ValueError("A captured electromagnetic potential snapshot is required")
    bounds = _bounds(display_bounds_m)
    sources = []
    for row in snapshot.contributions:
        sampling_bounds, support_planes = _sampling_hints(row.potential)
        sources.append(_PotentialFieldVectorSource(row.source_id, row.source_id, row.kind, row.owner_ids,
            bounds, type(row.potential).__name__,
            ("Fields come directly from this retained potential; the display box does not extend its validity.",
             f"Potential identity: {row.potential.digest}",
             "Sampling anchors do not truncate the potential; only declared finite faces block texture interpolation.",
             "Atomic source ownership: all listed physical owners belong to this one inseparable contribution."),
            sampling_bounds_m=sampling_bounds, support_planes_m=support_planes))
    return PotentialFieldVectorScene(tuple(sources), bounds, (
        f"Executed-potential snapshot: {snapshot.numerical_digest}",
        "B = curl A and E = -grad phi use the same retained representations as the potential executors.",
        "Only explicitly captured sources are represented; omitted components and unknown domains are not inferred as vacuum.",
        "View controls do not recalculate fields or modify their captured operating point."), snapshot)
