"""Conservative dependencies on an already executed global swept box.

Only explicit compact exteriors and Gaussian binary64 zeros may disappear
from a dependency key. Native support is never used to truncate round fields.
The box must include all RK stages, scalar-potential and contact queries.
"""
import numpy as np
from collections import OrderedDict
from threading import RLock

from temsim.immutable_json import json_digest


SCHEMA = "unified-executed-spatial-dependencies-v1"
_ZERO_BOXES = OrderedDict()
_ZERO_LOCK = RLock()


def checked_bounds(bounds_m):
    bounds = np.asarray(bounds_m, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
        raise ValueError("Executed spatial bounds must be finite ordered (2, 3) metres")
    return bounds


def local_bounds(registration, bounds_m):
    """Outward-rounded enclosure of the exact local affine transformation."""
    bounds = checked_bounds(bounds_m)
    low = np.nextafter(bounds[0]-registration.origin_array_m, -np.inf)
    high = np.nextafter(bounds[1]-registration.origin_array_m, np.inf)
    lower, upper = np.zeros(3), np.zeros(3)
    rotation = registration.rotation_array
    with np.errstate(over="ignore", invalid="ignore"):
        for axis in range(3):
            for dimension in range(3):
                a, b = low[axis]*rotation[axis, dimension], high[axis]*rotation[axis, dimension]
                lower[dimension] = np.nextafter(lower[dimension]+np.nextafter(min(a, b), -np.inf), -np.inf)
                upper[dimension] = np.nextafter(upper[dimension]+np.nextafter(max(a, b), np.inf), np.inf)
    if not np.isfinite((lower, upper)).all():
        raise ValueError("Executed spatial transform exceeds finite interval arithmetic")
    return np.stack((lower, upper))


def _outside_slab(registration, interval_mm, bounds_m):
    local = local_bounds(registration, bounds_m)
    limits = np.asarray(interval_mm, dtype=float)*.001
    lower, upper = np.nextafter(limits[0], -np.inf), np.nextafter(limits[1], np.inf)
    return bool(local[1, 2] < lower or local[0, 2] > upper)


def potential_is_zero_on_box(potential, bounds_m):
    """Prove an admitted exact machine zero; unsupported proofs return False."""
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.dynamic_potential import DynamicDipolePotential
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.gaussian_remote_domain import POLICY, remote_certificate
    if type(potential) is DynamicDipolePotential:
        potential = potential.base
    if type(potential) is AnalyticPotentialDomain and not potential._round:
        field = potential.base.field
        return _outside_slab(field.registration, field.native_support_mm, bounds_m)
    if type(potential) is not HighOrderRoundLensPotential or potential.axial_domain != "gaussian_all_z":
        return False
    bounds = checked_bounds(bounds_m)
    identity = potential.digest
    with _ZERO_LOCK:
        for cached in _ZERO_BOXES.get(identity, ()):
            if np.all(bounds[0] >= cached[0]) and np.all(bounds[1] <= cached[1]):
                _ZERO_BOXES.move_to_end(identity)
                return True
    local = local_bounds(potential.registration, bounds_m)
    transverse = np.nextafter(np.maximum(abs(local[0, :2]), abs(local[1, :2])), np.inf)
    radius = np.nextafter(np.hypot(*transverse), np.inf)
    in_radius = radius <= potential.valid_radius_m
    if not in_radius and potential.remote_domain != POLICY:
        return False
    # Also require the primitive Gaussian itself to underflow. This excludes a
    # tiny mathematical bound whose intermediate rounded evaluation is nonzero.
    for amplitude, centre, sigma in potential.native.terms_t_m:
        if amplitude == 0.:
            continue
        separation = np.nextafter(max(0., local[0, 2]-centre, centre-local[1, 2]), 0.)
        with np.errstate(under="ignore", over="ignore"):
            u = np.nextafter(separation/sigma, 0.)
            # Stay well beyond the libm subnormal rounding boundary. The
            # provider's multiplication order is retained for this extra gate.
            if u <= 40. or np.exp(-.5*u*u) != 0.:
                return False
    # The existing remote proof bounds every retained polynomial and derivative
    # strictly below half the smallest subnormal; it does not zero sampled data.
    certificate = remote_certificate(potential.native, np.array([radius, 0., 0.]),
        potential.radial_order, _z_bounds=(local[0, 2], local[1, 2]))
    if not bool(certificate["admitted"]):
        return False
    with _ZERO_LOCK:
        retained = [box for box in _ZERO_BOXES.get(identity, ())
                    if not (np.all(box[0] >= bounds[0]) and np.all(box[1] <= bounds[1]))]
        _ZERO_BOXES[identity] = (*retained[-3:], bounds.copy())
        _ZERO_BOXES.move_to_end(identity)
        while len(_ZERO_BOXES) > 512:
            _ZERO_BOXES.popitem(last=False)
    return True


def spatial_runtime_payload(runtime, bounds_m):
    """Key all field and opaque-slab dependencies consumed inside the box.

    Ownership order is retained. Unrelated scene metadata is not a field law.
    Tilted slabs are tested in their full local slab, never just their bore.
    """
    bounds = checked_bounds(bounds_m)
    fields = []
    from temsim.physics.dynamic_potential import DynamicDipolePotential
    for row in runtime.snapshot.contributions:
        # Time-domain guards are executed for every dynamic drive even outside
        # its spatial support. Retain its full identity so a changed flyback
        # cannot bypass the original executor's interval validation.
        if type(row.potential) is DynamicDipolePotential or not potential_is_zero_on_box(row.potential, bounds):
            fields.append((row.source_id, tuple(row.owner_ids), row.kind, row.potential.digest))
    geometry = []
    from temsim.physics.potential_geometry import _piece_dict
    for host in runtime.geometry.boundaries:
        pieces = [piece for piece in host.pieces if not _outside_slab(piece.registration,
            (piece.local_start_z_mm, piece.local_end_z_mm), bounds)]
        if pieces:
            geometry.append((host.host_key, tuple(host.source_keys), tuple(_piece_dict(piece) for piece in pieces)))
    return {"schema": SCHEMA, "fields": fields, "geometry": geometry,
            "gauge": runtime.snapshot.gauge.digest,
            "emission_time_s": runtime.metadata.get("emission_time_s", 0.)}


def spatial_runtime_identity(runtime, bounds_m):
    return json_digest(spatial_runtime_payload(runtime, bounds_m))
