"""Lossless snapshots of supported, already executed electrostatic fields.

Coordinates are global XYZ metres, Phi is the rise from the original cathode
in volts, and E = -grad(Phi) in V/m. Ground-referenced voltage, where declared,
is Phi plus ``ground_reference_offset_v``. The original interpolation basis is
retained: no mesh, solve, field fit, source, or wave admission is introduced.

Only the scalar field is captured. Material interception and launch geometry
remain separate contracts. First derivatives are piecewise cell derivatives;
second derivatives and Maxwell certification are deliberately not claimed.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from temsim.immutable_json import canonical_json_bytes, freeze_json, thaw_json
from .axis_field_interpolation import _evaluate
from .axis_regular_potential import AxisRegularPotential
from .axisymmetric_cut_field import AxisymmetricCutField
from .closed_gun_field import ClosedGunField
from .continuous_gun_field import ContinuousGunField
from .grounded_tip_field import GroundedTipField
from .instrument_electric import InstrumentElectricField
from .planar_gun_field import PlanarGunField


SCHEMA = "electric-potential-snapshot-v1"
_BILINEAR = "axisymmetric-r2-z-bilinear-v1"
_CUT = "axisymmetric-r2-z-cut-p1-regular-core-v1"
_FEM_NAMES = ("r", "z", "nodal_voltage", "cut_cells", "lookup", "origin",
              "inverse", "gradient", "phi0")
_METHODS = {
    PlanarGunField: ("interpolate", "potential_v_at_global_positions",
                    "potential_rise_v_at_global_positions", "field_at_global_positions_v_per_m"),
    ClosedGunField: ("interpolate", "potential_v_at_global_positions",
                    "potential_rise_v_at_global_positions", "field_at_global_positions_v_per_m"),
    ContinuousGunField: ("interpolate", "potential_v_at_global_positions",
                        "potential_rise_v_at_global_positions", "field_at_global_positions_v_per_m"),
    GroundedTipField: ("_interpolate", "potential_v_at_global_positions",
                      "potential_rise_v_at_global_positions", "field_at_global_positions_v_per_m"),
    AxisymmetricCutField: ("interpolate",),
    AxisRegularPotential: ("interpolate",),
    InstrumentElectricField: ("interpolate", "potential_rise_v_at_global_positions",
                              "field_at_global_positions_v_per_m"),
}
_ORIGINAL_METHODS = {cls: {name: getattr(cls, name) for name in names}
                     for cls, names in _METHODS.items()}


def _domain(r, z, continuation, bounds):
    kind = continuation["kind"]
    return {"radius_m": float(r[-1]) if bounds is None else float(bounds[1][0]),
            "z_min_m": (None if kind == "flat_cathode_below_zero" else float(z[0]))
                       if bounds is None else float(bounds[0][2]),
            "z_max_m": (None if kind == "grounded_downstream" else float(z[-1]))
                       if bounds is None else float(bounds[1][2]),
            "mesh_z_interval_m": [float(z[0]), float(z[-1])], "outside_domain": "reject"}


def _check_provider(provider):
    cls = type(provider)
    if cls not in _ORIGINAL_METHODS:
        raise ValueError(f"Unsupported electric potential provider: {cls.__module__}.{cls.__qualname__}")
    for name, original in _ORIGINAL_METHODS[cls].items():
        if (getattr(cls, name) is not original
                or getattr(getattr(provider, name), "__func__", None) is not original):
            raise ValueError(f"Overridden electric potential method is unsupported: {name}")
    if cls is ClosedGunField and PlanarGunField.interpolate is not _ORIGINAL_METHODS[PlanarGunField]["interpolate"]:
        raise ValueError("Overridden planar interpolation is unsupported")


@dataclass(frozen=True, slots=True)
class _Buffer:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def array(self):
        # A bytes owner prevents write access even through setflags(write=True).
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)

    @classmethod
    def capture(cls, name, value, dtype="<f8"):
        original = np.asarray(value)
        if dtype == "|b1" and original.dtype.kind != "b":
            raise ValueError(f"{name} must contain Boolean cell flags")
        if dtype == "<i8" and original.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer element indices")
        if dtype == "<f8" and (original.dtype.kind != "f" or original.dtype.itemsize != 8):
            raise ValueError(f"{name} requires the native float64 interpolation coefficients")
        if dtype == "<i8" and original.dtype.kind == "u" and np.any(original > np.iinfo(np.int64).max):
            raise ValueError(f"{name} contains unrepresentable element indices")
        array = np.ascontiguousarray(original, dtype=dtype)
        return cls(name, dtype, array.shape, array.tobytes())


@dataclass(frozen=True, slots=True)
class ElectricPotentialSnapshot:
    """Detached scalar-potential coefficients and their declared domain.

    Use ``from_provider`` for already constructed providers, or ``from_dict``
    for checked persistence. Neither path reads files or executes a solve.
    The public metadata is immutable and ``arrays`` returns immutable views.
    ``sample`` accepts (..., 3) and returns Phi (...) and E (..., 3).
    """

    metadata: Mapping
    _buffers: tuple[_Buffer, ...] = field(repr=False)
    digest: str = field(init=False)

    def __post_init__(self):
        metadata = freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise ValueError("Electric potential metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)
        buffers = tuple(self._buffers)
        if not all(type(item) is _Buffer and type(item.data) is bytes for item in buffers):
            raise ValueError("Electric potential coefficients must be immutable bytes")
        if len({item.name for item in buffers}) != len(buffers):
            raise ValueError("Duplicate electric potential arrays")
        object.__setattr__(self, "_buffers", tuple(sorted(buffers, key=lambda item: item.name)))
        self._validate()
        digest = sha256(canonical_json_bytes({"schema": SCHEMA, "metadata": metadata}))
        for item in self._buffers:
            digest.update(canonical_json_bytes({"name": item.name, "dtype": item.dtype, "shape": item.shape}))
            digest.update(item.data)
        object.__setattr__(self, "digest", digest.hexdigest())

    @property
    def payload(self):
        """Immutable physical/interpolation metadata; array bytes are separate."""
        return self.metadata

    @property
    def arrays(self):
        return {item.name: item.array() for item in self._buffers}

    def _validate(self):
        m = self.metadata
        required = {"basis", "provider_type", "potential_reference", "ground_reference_offset_v",
                    "coordinates", "units", "derivative_order", "derivative_semantics",
                    "continuation", "domain", "instrument_bounds_m", "request", "source_identities"}
        if set(m) != required or m["basis"] not in {_BILINEAR, _CUT}:
            raise ValueError("Unsupported electric potential metadata or interpolation basis")
        native_types = {cls.__module__+"."+cls.__qualname__ for cls in _METHODS
                        if cls is not InstrumentElectricField}
        if m["provider_type"] not in native_types:
            raise ValueError("Unsupported captured electric potential provider type")
        if (m["potential_reference"] != "rise_relative_to_original_tip_v"
                or m["coordinates"] != "global_xyz_metres"
                or m["units"] != {"phi": "V", "electric": "V/m"}
                or type(m["derivative_order"]) is not int or m["derivative_order"] != 1
                or m["derivative_semantics"] != "piecewise_cell_exact_gradient"):
            raise ValueError("Unsupported electric potential reference or derivative contract")
        for item in self._buffers:
            expected_dtype = "|b1" if item.name == "cut_cells" else "<i8" if item.name == "lookup" else "<f8"
            if (item.dtype != expected_dtype
                    or any(type(size) is not int or size < 0 for size in item.shape)
                    or len(item.data) != int(np.prod(item.shape))*np.dtype(item.dtype).itemsize):
                raise ValueError("Invalid electric potential array representation")
        a = self.arrays
        expected = {"r", "z", "voltage"} if m["basis"] == _BILINEAR else set(_FEM_NAMES) | {"core_s", "core_slope"}
        if set(a) != expected or not all(np.isfinite(value).all() for value in a.values()):
            raise ValueError("Invalid electric potential coefficient arrays")
        r, z = a["r"], a["z"]
        if (r.ndim != 1 or z.ndim != 1 or len(r) < 2 or len(z) < 2 or r[0] != 0
                or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)):
            raise ValueError("Electric potential axes must be finite and strictly increasing from r=0")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            radial_widths = np.diff(r*r)
            axial_widths = np.diff(z)
        if (not np.isfinite(radial_widths).all() or np.any(radial_widths <= 0)
                or not np.isfinite(axial_widths).all()):
            raise ValueError("Electric potential cells must have finite resolved widths")
        voltage = a["voltage"] if m["basis"] == _BILINEAR else a["nodal_voltage"]
        if voltage.shape != (len(r), len(z)):
            raise ValueError("Electric potential grid shape mismatch")
        if m["basis"] == _CUT:
            n = len(a["phi0"])
            shapes = {"cut_cells": (len(r)-1, len(z)-1), "lookup": (2*(len(r)-1)*(len(z)-1), 2),
                      "origin": (n, 2), "inverse": (n, 2, 2), "gradient": (n, 2), "phi0": (n,),
                      "core_s": (len(z),), "core_slope": (len(z),)}
            if (any(a[key].shape != shape for key, shape in shapes.items())
                    or a["cut_cells"].dtype.kind != "b" or a["lookup"].dtype.kind != "i"
                    or np.any(a["lookup"] < -1) or np.any(a["lookup"] >= n)
                    or np.any(a["core_s"] < 0) or np.any(a["core_s"] > r[-1]**2)):
                raise ValueError("Invalid cut-cell or regular-axis potential representation")
        continuation = m["continuation"]
        if not isinstance(continuation, Mapping) or set(continuation) != {"kind", "value_v"}:
            raise ValueError("Invalid electric potential continuation")
        kind = continuation["kind"]
        if kind not in {"none", "flat_cathode_below_zero", "grounded_downstream"}:
            raise ValueError("Unsupported electric potential continuation")
        if kind == "none" and continuation["value_v"] is not None:
            raise ValueError("Finite-domain potential must not imply a continuation")
        if kind == "flat_cathode_below_zero" and (z[0] != 0 or continuation["value_v"] != 0):
            raise ValueError("Flat cathode continuation requires the z=0 zero-rise boundary")
        if kind == "grounded_downstream" and not isinstance(continuation["value_v"], (int, float)):
            raise ValueError("Grounded continuation requires an explicit finite high tension")
        offset = m["ground_reference_offset_v"]
        if offset is not None and (isinstance(offset, bool) or not isinstance(offset, (int, float))):
            raise ValueError("Invalid ground-reference offset")
        if kind == "grounded_downstream" and offset != -continuation["value_v"]:
            raise ValueError("Grounded continuation and potential reference disagree")
        bounds = m["instrument_bounds_m"]
        if bounds is not None:
            bounds = np.asarray(bounds, float)
            if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
                    or np.any(bounds[0] >= bounds[1]) or bounds[1, 0] > r[-1]
                    or not np.array_equal(bounds[:, :2], [[-bounds[1, 0]]*2, [bounds[1, 0]]*2])
                    or (kind != "flat_cathode_below_zero" and bounds[0, 2] < z[0])
                    or (kind != "grounded_downstream" and bounds[1, 2] > z[-1])):
                raise ValueError("Instrument domain exceeds the captured scalar-field contract")
        if thaw_json(m["domain"]) != _domain(r, z, continuation, bounds):
            raise ValueError("Electric potential domain metadata disagrees with its coefficients")

    @classmethod
    def from_provider(cls, provider):
        """Capture known native representations without querying or solving them."""
        _check_provider(provider)
        bounds, identities = None, {}
        if type(provider) is InstrumentElectricField:
            if (provider.provider is not provider.base_field
                    or getattr(provider.provider, "wien_field", None) is not None):
                raise ValueError("Composed instrument electric providers require an explicit snapshot adapter")
            bounds = np.asarray(provider.bounds_m, float).tolist()
            identities = {"physical": provider.physical_identity, "numerical": provider.numerical_identity,
                          "request": provider.request_identity}
            provider = provider.base_field
            _check_provider(provider)
        kind = type(provider)
        continuation, offset = {"kind": "none", "value_v": None}, None
        arrays = {}
        if kind in {PlanarGunField, ClosedGunField}:
            basis = _BILINEAR
            arrays = {key: getattr(provider, key) for key in ("r", "z", "voltage")}
            if kind is ClosedGunField:
                continuation = {"kind": "flat_cathode_below_zero", "value_v": 0.}
        else:
            basis = _CUT
            if kind in {ContinuousGunField, GroundedTipField}:
                fem = getattr(provider, "_fem", None)
                regular = getattr(provider, "_regular", None)
                if kind is ContinuousGunField and regular is None:
                    raise ValueError("Continuous gun requires its executed regular-axis interpolation")
                if kind is GroundedTipField:
                    offset = -float(provider.high_tension_v)
                    continuation = {"kind": "grounded_downstream", "value_v": -offset}
                    if fem is None:
                        raise ValueError("Historical grounded bilinear fixtures are not supported by this adapter")
            elif kind is AxisRegularPotential:
                regular, fem = provider, provider.field
            elif kind is AxisymmetricCutField:
                regular, fem = None, provider
            else:
                raise ValueError("Unsupported nested electric potential provider")
            _check_provider(fem)
            if type(fem) is not AxisymmetricCutField:
                raise ValueError("Electric potential requires the native cut-cell representation")
            arrays = {key: getattr(fem, key) for key in _FEM_NAMES}
            arrays.update(core_s=np.zeros(len(fem.z)), core_slope=np.zeros(len(fem.z)))
            if regular is not None:
                _check_provider(regular)
                if type(regular) is not AxisRegularPotential or regular.field is not fem:
                    raise ValueError("Regular-axis interpolation must refer to the same captured FEM")
                arrays.update(core_s=regular.s, core_slope=regular.slope)
            if kind in {ContinuousGunField, GroundedTipField} and (
                    not np.array_equal(provider.r, fem.r) or not np.array_equal(provider.z, fem.z)):
                raise ValueError("Gun domain and cut-cell interpolation axes disagree")
        metadata = {"basis": basis, "provider_type": kind.__module__+"."+kind.__qualname__,
            "potential_reference": "rise_relative_to_original_tip_v", "ground_reference_offset_v": offset,
            "coordinates": "global_xyz_metres", "units": {"phi": "V", "electric": "V/m"},
            "derivative_order": 1, "derivative_semantics": "piecewise_cell_exact_gradient",
            "continuation": continuation, "instrument_bounds_m": bounds,
            "domain": _domain(arrays["r"], arrays["z"], continuation, bounds),
            "request": getattr(provider, "request", None), "source_identities": identities}
        buffers = tuple(_Buffer.capture(name, value, "|b1" if name == "cut_cells" else
                        "<i8" if name == "lookup" else "<f8") for name, value in arrays.items())
        return cls(metadata, buffers)

    def sample(self, positions_m):
        """Return Phi and E from the same scalar law; reject undeclared domains."""
        p = np.asarray(positions_m, dtype=float)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError("Electric potential positions must be finite XYZ metres with shape (..., 3)")
        q = p.reshape(-1, 3).copy()
        a, m = self.arrays, self.metadata
        r, z = a["r"], a["z"]
        bounds = m["instrument_bounds_m"]
        radius = np.hypot(q[:, 0], q[:, 1])
        if bounds is not None:
            bounds = np.asarray(bounds)
            if np.any(q < bounds[0]) or np.any(q > bounds[1]) or np.any(radius > bounds[1, 0]):
                raise ValueError("Position outside the fixed instrument electric domain")
        kind = m["continuation"]["kind"]
        extended = q[:, 2] < 0 if kind == "flat_cathode_below_zero" else q[:, 2] >= z[-1]
        if kind == "flat_cathode_below_zero":
            q[:, 2] = np.maximum(q[:, 2], 0.)
        elif kind == "grounded_downstream":
            q[:, 2] = np.minimum(q[:, 2], z[-1])
        if np.any(radius > r[-1]) or np.any(q[:, 2] < z[0]) or np.any(q[:, 2] > z[-1]):
            raise ValueError("Position outside captured electric potential domain; no extrapolation")
        if m["basis"] == _BILINEAR:
            phi, electric = _bilinear(q, r, z, a["voltage"])
        else:
            # The existing stateless scalar evaluator retains cut triangles and
            # regular-axis cells without reconstructing a mutable provider.
            phi, electric = _evaluate(q, r, z, a["nodal_voltage"], a["cut_cells"], a["lookup"],
                a["origin"], a["inverse"], a["gradient"], a["phi0"], a["core_s"], a["core_slope"])
        if kind != "none":
            phi[extended] = m["continuation"]["value_v"]
            electric[extended] = 0.
        return {"phi_v": phi.reshape(p.shape[:-1]), "E_v_per_m": electric.reshape(p.shape)}

    def to_dict(self):
        return {"schema": SCHEMA, "metadata": thaw_json(self.metadata), "digest": self.digest,
            "arrays": {item.name: {"dtype": item.dtype, "shape": list(item.shape),
                "base64": base64.b64encode(item.data).decode("ascii")} for item in self._buffers}}

    @classmethod
    def from_dict(cls, record):
        if (not isinstance(record, Mapping) or set(record) != {"schema", "metadata", "arrays", "digest"}
                or record["schema"] != SCHEMA or not isinstance(record["arrays"], Mapping)):
            raise ValueError("Unsupported electric potential snapshot schema")
        buffers = []
        try:
            for name, item in record["arrays"].items():
                if set(item) != {"dtype", "shape", "base64"}:
                    raise ValueError("Unexpected electric potential array fields")
                buffers.append(_Buffer(name, item["dtype"], tuple(item["shape"]),
                    base64.b64decode(item["base64"], validate=True)))
            snapshot = cls(record["metadata"], tuple(buffers))
        except (TypeError, KeyError, OverflowError) as error:
            raise ValueError("Invalid electric potential snapshot representation") from error
        if snapshot.digest != record["digest"]:
            raise ValueError("Electric potential snapshot digest mismatch")
        return snapshot


def _bilinear(p, r, z, voltage):
    """Native planar arithmetic, retaining scalar/batch boundary conventions."""
    radius = np.hypot(p[:, 0], p[:, 1])
    i = np.clip(np.searchsorted(r, radius, side="right")-1, 0, len(r)-2)
    j = np.clip(np.searchsorted(z, p[:, 2], side="right")-1, 0, len(z)-2)
    ds, dz = r[i+1]**2-r[i]**2, z[j+1]-z[j]
    u, v = (radius**2-r[i]**2)/ds, (p[:, 2]-z[j])/dz
    a, b, c, d = voltage[i, j], voltage[i+1, j], voltage[i, j+1], voltage[i+1, j+1]
    bottom, top = a+u*(b-a), c+u*(d-c)
    phi = bottom+v*(top-bottom)
    derivative_s = ((1-v)*(b-a)+v*(d-c))/ds
    electric = np.empty_like(p)
    electric[:, :2] = -2*p[:, :2]*derivative_s[:, None]
    electric[:, 2] = -(top-bottom)/dz
    return phi, electric


__all__ = ("ElectricPotentialSnapshot",)
