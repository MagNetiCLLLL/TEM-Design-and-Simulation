"""Strict CPU contact acceleration for the existing absorbing material law.

Unrotated circular primitives use the same slab/cylinder calculation in a
compiled loop. Rotated, unknown and numerically ambiguous boundary cases keep
the authoritative NumPy primitive. No geometric tolerance enlarges a hole.
"""
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import numpy as np

from temsim.physics.posed_wave_aperture import PosedWaveAperture
from temsim.physics.posed_wave_hardware import PosedWaveBore

try:
    from numba import njit
except ImportError:  # Optional acceleration never changes the material law.
    njit = None


def _native_contact_impl(start, end, origins, lower, upper, radius, aperture,
                         native, selected):
    count = len(start)
    fractions = np.full(count, np.inf)
    owners = np.full(count, -1, dtype=np.int64)
    ambiguous = np.zeros(len(native), dtype=np.bool_)
    for position in range(len(selected)):
        piece = selected[position]
        if not native[piece]:
            continue
        # The scalar hypot implementation can differ by its last bit from a
        # NumPy vector ufunc. Delegate every nearby entrance edge unchanged.
        for ray in range(count):
            x = start[ray, 0]-origins[piece, 0]
            y = start[ray, 1]-origins[piece, 1]
            z = start[ray, 2]-origins[piece, 2]
            if lower[piece] <= z <= upper[piece]:
                radial = np.hypot(x, y)
                scale = max(radial, radius[piece], np.finfo(np.float64).tiny)
                if abs(radial-radius[piece]) <= 32*np.finfo(np.float64).eps*scale:
                    ambiguous[piece] = True
                    break
        if ambiguous[piece]:
            continue
        for ray in range(count):
            x = start[ray, 0]-origins[piece, 0]
            y = start[ray, 1]-origins[piece, 1]
            z = start[ray, 2]-origins[piece, 2]
            dx = end[ray, 0]-start[ray, 0]
            dy = end[ray, 1]-start[ray, 1]
            dz = end[ray, 2]-start[ray, 2]
            moving = abs(dz) > 1e-14
            safe_dz = dz if moving else 1.
            a = (lower[piece]-z)/safe_dz
            b = (upper[piece]-z)/safe_dz
            first = max(0., min(a, b)) if moving else 0.
            last = min(1., max(a, b)) if moving else 1.
            inside = lower[piece] <= z <= upper[piece]
            active = last >= first if moving else inside
            entry_x, entry_y = x+first*dx, y+first*dy
            exit_x, exit_y = x+last*dx, y+last*dy
            radius2 = radius[piece]*radius[piece]
            entry2 = entry_x*entry_x+entry_y*entry_y
            entry_out = entry2 > radius2
            exit_out = exit_x*exit_x+exit_y*exit_y > radius2
            qa = dx*dx+dy*dy
            qb = 2*(entry_x*dx+entry_y*dy)
            qc = entry2-radius2
            root = (-qb+np.sqrt(max(0., qb*qb-4*qa*qc)))/(2*qa if qa > 0 else 1.)
            hit = first if entry_out or radius[piece] == 0 else first+root
            trial = min(max(hit, first), last) if active and (entry_out or exit_out or radius[piece] == 0) else np.inf
            radial = np.hypot(x, y)
            entrance_passed = (radius[piece] > 0 and radial <= radius[piece]) if aperture[piece] else radial < radius[piece]
            if inside and not entrance_passed:
                trial = 0.
            if trial < fractions[ray]:
                fractions[ray] = trial
                owners[ray] = piece
    return fractions, owners, ambiguous


_native_contact = (njit(cache=True, fastmath=False, error_model="numpy")(_native_contact_impl)
                   if njit is not None else None)
_CACHE = OrderedDict()
_LOCK = RLock()
_MAX_CACHED_GEOMETRIES = 32


def _frozen(values, dtype=float):
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class ActiveContactPieces:
    geometry: object
    indices: np.ndarray

    def __iter__(self):
        return (self.geometry.rows[int(index)] for index in self.indices)

    def __len__(self):
        return len(self.indices)


@dataclass(frozen=True, slots=True)
class CompiledContactGeometry:
    rows: tuple
    origins_mm: np.ndarray
    lower_mm: np.ndarray
    upper_mm: np.ndarray
    radii_mm: np.ndarray
    aperture: np.ndarray
    native: np.ndarray
    axial_lower_m: np.ndarray
    axial_upper_m: np.ndarray

    def active(self, lower_m, upper_m):
        # A tilted slab is not bounded by the opening's global-Z extent. Its
        # bounds stay infinite, just as in the previous active-piece selector.
        selected = ~((upper_m < self.axial_lower_m) | (lower_m > self.axial_upper_m))
        return ActiveContactPieces(self, _frozen(np.flatnonzero(selected), np.int64))

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


def _compile_rows(rows):
    rows = tuple(rows)
    count = len(rows)
    origins = np.zeros((count, 3))
    lower, upper, radii = np.zeros(count), np.zeros(count), np.ones(count)
    aperture, native = np.zeros(count, bool), np.zeros(count, bool)
    axial_lower, axial_upper = np.full(count, -np.inf), np.full(count, np.inf)
    for index, (_key, piece) in enumerate(rows):
        # Unknown and subclassed primitives may override their material rule.
        if type(piece) not in (PosedWaveAperture, PosedWaveBore):
            continue
        rotation = piece.registration.rotation_array
        origin_m = piece.registration.origin_array_m
        origins[index] = origin_m*1000
        lower[index], upper[index], radii[index] = piece.local_start_z_mm, piece.local_end_z_mm, piece.radius_mm
        aperture[index] = type(piece) is PosedWaveAperture
        known = np.r_[origins[index], lower[index], upper[index], radii[index]]
        # Large finite values retain NumPy's overflow/nonfinite semantics.
        native[index] = (np.array_equal(rotation, np.eye(3))
                         and np.isfinite(known).all() and np.max(np.abs(known)) < 1e50)
        if np.array_equal(rotation[2], (0., 0., 1.)) and np.array_equal(rotation[:, 2], (0., 0., 1.)):
            axial_lower[index] = lower[index]*.001+origin_m[2]
            axial_upper[index] = upper[index]*.001+origin_m[2]
    return CompiledContactGeometry(rows, *(_frozen(value) for value in (origins, lower, upper, radii)),
                                   _frozen(aperture, bool), _frozen(native, bool),
                                   _frozen(axial_lower), _frozen(axial_upper))


def compile_contact_geometry(geometry):
    """Compile frozen boundary registration once, retaining original hosts."""
    from temsim.physics.potential_geometry import PotentialGeometrySnapshot
    if type(geometry) is not PotentialGeometrySnapshot:
        return _compile_rows((host.host_key, piece) for host in geometry.boundaries for piece in host.pieces)
    key = geometry.digest
    with _LOCK:
        previous = _CACHE.get(key)
        if previous is not None:
            _CACHE.move_to_end(key)
            return previous
    result = _compile_rows((host.host_key, piece) for host in geometry.boundaries for piece in host.pieces)
    with _LOCK:
        result = _CACHE.setdefault(key, result)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_CACHED_GEOMETRIES:
            _CACHE.popitem(last=False)
    return result


def contact(pieces, before, after):
    """Return unchanged first-contact fractions and physical-host labels."""
    if isinstance(pieces, ActiveContactPieces):
        geometry, selected = pieces.geometry, pieces.indices
    else:
        geometry = pieces if isinstance(pieces, CompiledContactGeometry) else _compile_rows(pieces)
        selected = np.arange(len(geometry.rows), dtype=np.int64)
    before, after = np.asarray(before), np.asarray(after)
    if before.dtype != np.dtype(float) or after.dtype != np.dtype(float):
        # Retain the original input-dtype multiplication before the primitive
        # promotes its local-coordinate calculation. Promoting float32 first
        # would move a mechanical boundary by its input rounding error.
        fractions = np.full(len(before), np.inf)
        labels = np.full(len(before), "", dtype=object)
        for index in selected:
            key, piece = geometry.rows[int(index)]
            trial = PosedWaveAperture.first_contact_fraction(piece, before*1000, after*1000)
            outside = ~piece.transmission_mask(before[:, 0]*1000, before[:, 1]*1000, before[:, 2]*1000)
            trial = np.where(outside, 0., trial)
            earlier = trial < fractions
            fractions[earlier], labels[earlier] = trial[earlier], key
        return fractions, labels
    start, end = before*1000, after*1000
    safe = (np.isfinite(start).all() and np.isfinite(end).all()
            and (not start.size or max(np.max(np.abs(start)), np.max(np.abs(end))) < 1e50))
    if _native_contact is not None and safe and np.any(geometry.native[selected]):
        fractions, owners, ambiguous = _native_contact(start, end, geometry.origins_mm,
            geometry.lower_mm, geometry.upper_mm, geometry.radii_mm, geometry.aperture,
            geometry.native, selected)
        fallback = selected[~geometry.native[selected] | ambiguous[selected]]
    else:
        fractions = np.full(len(before), np.inf)
        owners = np.full(len(before), -1, dtype=np.int64)
        fallback = selected
    for index in fallback:
        _key, piece = geometry.rows[int(index)]
        trial = PosedWaveAperture.first_contact_fraction(piece, start, end)
        outside = ~piece.transmission_mask(start[:, 0], start[:, 1], start[:, 2])
        trial = np.where(outside, 0., trial)
        earlier = (trial < fractions) | ((trial == fractions) & np.isfinite(trial) & (index < owners))
        fractions[earlier], owners[earlier] = trial[earlier], index
    labels = np.full(len(before), "", dtype=object)
    for index in np.unique(owners[owners >= 0]):
        labels[owners == index] = geometry.rows[int(index)][0]
    return fractions, labels
