"""Frozen native scan laws with one consistent prescribed A(r,t) and E(r,t).

The spatial model remains the existing compact uniform native dipole. Its
prescribed time dependence is not a coil circuit, fringe or radiation solve.
Faraday induction is retained as E_induced=-partial_t A, and a raster flyback
is an explicit discontinuity: ordinary derivatives cannot integrate across it.
All time arguments are absolute laboratory seconds. No epoch is added here.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
import math

import numpy as np
from scipy.constants import e

from temsim.immutable_json import json_digest
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, _canonical_owner


SCHEMA = "native-dynamic-potential-snapshot-v1"


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (ValueError,OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty identifier")
    return value


def _times(time_s):
    value = np.asarray(time_s)
    if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
        raise ValueError("Dynamic potentials require finite absolute laboratory times in seconds")
    return np.asarray(value, dtype=float)


@dataclass(frozen=True, slots=True)
class NativeTransverseDrive:
    """One logical command after its captured physical-coil coupling matrix."""

    key: str
    kind: str
    matrix_mrad: tuple
    period_s: float
    lines: int = 1
    phase_rad: float = 0.

    def __post_init__(self):
        _identifier(self.key, "Logical drive key")
        if self.kind not in ("raster", "sine"):
            raise ValueError("Dynamic native drive must be raster or sine")
        matrix = np.asarray(self.matrix_mrad)
        if matrix.dtype.kind not in "iuf" or matrix.shape != (2, 2) or not np.isfinite(matrix).all():
            raise ValueError("Coil command matrix must be a finite real 2x2 array")
        if isinstance(self.lines, (bool, np.bool_)) or not isinstance(self.lines, (int, np.integer)) or self.lines < 1:
            raise ValueError("Raster line count must be a positive integer")
        object.__setattr__(self, "matrix_mrad", tuple(tuple(float(v) for v in row) for row in matrix))
        object.__setattr__(self, "period_s", _number(self.period_s, "Drive period", positive=True))
        object.__setattr__(self, "phase_rad", _number(self.phase_rad, "Drive phase"))
        object.__setattr__(self, "lines", int(self.lines))
        if self.kind == "sine" and np.any(matrix[:, 1] != 0.):
            raise ValueError("Sine drive uses its first matrix column only")
        for order in range(4):
            if not np.isfinite(self.derivative_bound(order)).all():
                raise ValueError("Drive derivatives exceed the finite numerical range")

    @property
    def varying(self):
        matrix = np.asarray(self.matrix_mrad)
        return bool(np.any(matrix[:, 0] != 0.) or
                    (self.kind == "raster" and self.lines > 1 and np.any(matrix[:, 1] != 0.)))

    def _line_interval(self, time_s):
        """One authoritative right-continuous binary64 line partition.

        Boundaries are rounded integer multiples of the retained line period.
        Quotient rounding is corrected by comparing against those boundaries;
        the waveform and the event guard never use different clock formulas.
        """
        time = _times(time_s)
        line_period = self.period_s/self.lines
        if (not math.isfinite(line_period) or line_period <= 0
                or np.any(np.spacing(np.abs(time)) >= .25*line_period)):
            raise ValueError("Raster time cannot resolve its declared line interval")
        index = np.floor(time/line_period)
        index = np.where(time < index*line_period,index-1,index)
        index = np.where(time >= (index+1)*line_period,index+1,index)
        lower,upper = index*line_period,(index+1)*line_period
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any((lower>time)|(upper<=time)|(upper<=lower))):
            raise ValueError("Raster event times exceed the resolved numerical range")
        return index,lower,upper

    def _raster(self, time_s):
        time = _times(time_s)
        index,lower,_ = self._line_interval(time)
        fraction = (time-lower)/(self.period_s/self.lines)
        row = np.remainder(index,self.lines)
        return 2.*fraction-1., 2.*(row+.5)/self.lines-1., time == lower

    def evaluate(self, time_s):
        """Return command and first two ordinary time derivatives in mrad."""
        time = _times(time_s)
        matrix = np.asarray(self.matrix_mrad)
        if self.kind == "raster":
            x, y, _ = self._raster(time)
            value = np.stack((x, y), axis=-1) @ matrix.T
            first = np.broadcast_to(matrix[:, 0]*(2./(self.period_s/self.lines)), value.shape).copy()
            return value, first, np.zeros_like(value)
        omega = 2.*math.pi/self.period_s
        phase = omega*time+self.phase_rad
        amplitude = matrix[:, 0]
        return (np.sin(phase)[..., None]*amplitude,
                (omega*np.cos(phase))[..., None]*amplitude,
                (-omega**2*np.sin(phase))[..., None]*amplitude)

    def derivative_bound(self, order):
        if isinstance(order, bool) or order not in (0, 1, 2, 3):
            raise ValueError("Drive derivative order must be zero through three")
        matrix = np.abs(np.asarray(self.matrix_mrad))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self.kind == "sine":
                return matrix[:, 0]*np.float64(2.*math.pi/self.period_s)**order
            if order == 0:
                return matrix.sum(axis=1)
            return matrix[:, 0]*(2./(self.period_s/self.lines)) if order == 1 else np.zeros(2)

    def require_differentiable_time(self, time_s):
        if self.kind == "raster" and self.varying and np.any(self._raster(time_s)[2]):
            raise ValueError(f"{self.key}: raster flyback has no ordinary time derivative; a temporal jump operator is required")

    def next_time_event(self, time_s):
        time = _number(time_s, "Event search time")
        if self.kind != "raster" or not self.varying:
            return math.inf
        return float(self._next_time_events(np.asarray(time)))

    def _next_time_events(self,time):
        """Use the same exact interval identity as the waveform evaluation."""
        return self._line_interval(time)[2]

    def require_time_interval(self, start_s, stop_s):
        start, stop = np.broadcast_arrays(_times(start_s), _times(stop_s))
        if np.any(stop < start):
            raise ValueError("Dynamic time interval must be nondecreasing")
        if self.kind == "raster" and self.varying:
            if np.any(self._next_time_events(start)<=stop):
                raise ValueError(f"{self.key}: interval crosses a raster flyback; induced temporal impulse is not modelled")

    def to_dict(self):
        return {"key": self.key, "kind": self.kind, "matrix_mrad": self.matrix_mrad,
                "period_s": self.period_s, "lines": self.lines, "phase_rad": self.phase_rad,
                "time_policy":"rounded-line-interval-v1"}

    @classmethod
    def from_dict(cls, record):
        if (not isinstance(record, Mapping) or set(record) != {"key", "kind", "matrix_mrad", "period_s", "lines", "phase_rad", "time_policy"}
                or record["time_policy"] != "rounded-line-interval-v1"):
            raise ValueError("Invalid frozen native drive record")
        return cls(**{key:value for key,value in record.items() if key != "time_policy"})


@dataclass(frozen=True, slots=True)
class DynamicDipolePotential:
    """One physical uniform coil, including all its logical scan commands."""

    base: AnalyticPotentialDomain
    owner_ids: tuple
    drives: tuple[NativeTransverseDrive, ...]
    reference_time_s: float
    digest: str = field(init=False)

    def __post_init__(self):
        if (type(self.base) is not AnalyticPotentialDomain or type(self.base.base.field) is not FrozenPosedMultipole
                or self.base.base.field.envelope_kind != "uniform"
                or any(x+y != 1 for x, y, _ in self.base.base.field.polynomial_terms)):
            raise ValueError("Dynamic native coil requires the retained uniform degree-one Az potential")
        owners = tuple(_identifier(value, "Physical coil owner") for value in self.owner_ids)
        if not owners or len(set(owners)) != len(owners):
            raise ValueError("Dynamic coil owners must be nonempty and unique")
        drives = tuple(self.drives)
        if not drives or any(type(value) is not NativeTransverseDrive for value in drives):
            raise ValueError("Dynamic coil requires exact frozen native drive laws")
        if len({drive.key for drive in drives}) != len(drives):
            raise ValueError("A logical drive cannot be counted twice in a physical coil")
        object.__setattr__(self, "base", AnalyticPotentialDomain.from_dict(self.base.to_dict()))
        object.__setattr__(self, "owner_ids", owners)
        object.__setattr__(self, "drives", drives)
        object.__setattr__(self, "reference_time_s", _number(self.reference_time_s, "Reference time"))
        _number(self.conversion_t_per_mrad,"Dynamic coil angular conversion")
        for row in self.temporal_bounds()["derivatives"]:
            if not np.isfinite((*row["coefficient_abs_bound"],row["A_norm_bound"],row["jac_A_norm_bound"])).all():
                raise ValueError("Dynamic coil derivative bounds exceed the finite numerical range")
        object.__setattr__(self, "digest", json_digest(self._payload()))

    @property
    def source_id(self):
        return self.base.source_id

    @property
    def potential(self):
        return self

    @property
    def kind(self):
        return "magnetic"

    @property
    def conversion_t_per_mrad(self):
        native = self.base.base.field
        length = (native.native_support_mm[1]-native.native_support_mm[0])*.001
        return native.reference_momentum/(-e)/length*.001

    def coefficients(self, time_s):
        time = _times(time_s)
        native = self.base.base.field
        reference = dict(((x, y), value) for x, y, value in native.polynomial_terms)
        coefficients = np.broadcast_to((reference.get((1, 0), 0.), reference.get((0, 1), 0.)), (*time.shape, 2)).copy()
        first, second = np.zeros_like(coefficients), np.zeros_like(coefficients)
        factor = self.conversion_t_per_mrad
        for drive in self.drives:
            value, dot, ddot = drive.evaluate(time)
            coefficients += factor*(value-drive.evaluate(self.reference_time_s)[0])
            first += factor*dot
            second += factor*ddot
        if any(not np.isfinite(value).all() for value in (coefficients, first, second)):
            raise ValueError("Dynamic coil coefficients exceed the finite numerical range")
        return coefficients, first, second

    def sample(self, positions_m, time_s, *, derivatives=False, _axial_side=None):
        if _axial_side not in (None,"inside","outside"):
            raise ValueError("An axial jet side must be inside or outside")
        native = self.base.base.field
        if _axial_side is not None:
            rotation = native.registration.rotation_array
            if not (np.array_equal(rotation[2],(0.,0.,1.)) and np.array_equal(rotation[:,2],(0.,0.,1.))):
                raise ValueError("One-sided compact axial jets require an exactly aligned global Z axis")
        points, active = self.base._positions(positions_m, derivatives=derivatives and _axial_side is None)
        if _axial_side == "outside":
            local_z = native.registration.positions_global_to_local_m(points)[...,2]
            lower,upper = np.asarray(native.native_support_mm)*.001
            active = active & (local_z != lower) & (local_z != upper)
        times = np.broadcast_to(_times(time_s), points.shape[:-1])
        selection = active.reshape(-1)
        output = {name: np.zeros(points.shape) for name in
                  ("A_t_m", "B_t", "E_v_per_m", "dA_dt_t_m_per_s", "d2A_dt2_t_m_per_s2")}
        if derivatives:
            output.update(jac_A_t=np.zeros((*points.shape, 3)),
                          hess_A_t_per_m=np.zeros((*points.shape, 3, 3)),
                          jac_dA_dt_t_per_s=np.zeros((*points.shape, 3)))
        if not selection.any():
            return output
        selected_time = times.reshape(-1)[selection]
        for drive in self.drives:
            drive.require_differentiable_time(selected_time)
        registration = self.base.base.field.registration
        local = registration.positions_global_to_local_m(points.reshape(-1, 3)[selection])
        rotation, axis = registration.rotation_array, registration.rotation_array[:, 2]
        values = self.coefficients(selected_time)
        for name, coefficients in zip(("A_t_m", "dA_dt_t_m_per_s", "d2A_dt2_t_m_per_s2"), values):
            output[name].reshape(-1, 3)[selection] = (np.sum(local[:, :2]*coefficients, axis=1)[:, None])*axis
        output["E_v_per_m"] = -output["dA_dt_t_m_per_s"]
        output["B_t"].reshape(-1, 3)[selection] = np.c_[values[0][:, 1], -values[0][:, 0], np.zeros(selection.sum())]@rotation.T
        if derivatives:
            for key, coefficients in (("jac_A_t", values[0]), ("jac_dA_dt_t_per_s", values[1])):
                gradient = np.c_[coefficients, np.zeros(selection.sum())]@rotation.T
                output[key].reshape(-1, 3, 3)[selection] = np.einsum("i,nj->nij", axis, gradient)
        if any(not np.isfinite(value).all() for value in output.values()):
            raise ValueError("Dynamic potential evaluation exceeds the finite numerical range")
        return output

    def sample_jet(self, positions_m, time_s, *, axial_side=None):
        """Retained linear Az jet; only declared aligned faces admit a side.

        Coordinates are unchanged. The default rejects two-sided derivatives
        on a hard face. Explicit inside/outside selects that model's one-sided
        limit only at exact compact faces; interior and exterior are unchanged.
        """
        return self.sample(positions_m, time_s, derivatives=True, _axial_side=axial_side)

    def require_time_interval(self, start_s, stop_s):
        for drive in self.drives:
            drive.require_time_interval(start_s, stop_s)

    def next_time_event(self, time_s):
        return min(drive.next_time_event(time_s) for drive in self.drives)

    def temporal_bounds(self):
        native = self.base.base.field
        coefficient = dict(((x, y), value) for x, y, value in native.polynomial_terms)
        constant = np.array((coefficient.get((1, 0), 0.), coefficient.get((0, 1), 0.)))
        for drive in self.drives:
            constant -= self.conversion_t_per_mrad*drive.evaluate(self.reference_time_s)[0]
        axis_norm = float(np.linalg.norm(native.registration.rotation_array[:, 2]))
        rotation_norm = float(np.linalg.norm(native.registration.rotation_array, ord=2))
        rows = []
        for order in range(4):
            bound = np.abs(constant) if order == 0 else np.zeros(2)
            for drive in self.drives:
                bound += abs(self.conversion_t_per_mrad)*drive.derivative_bound(order)
            norm = float(np.linalg.norm(bound))
            rows.append({"order": order, "coefficient_abs_bound": bound.tolist(),
                         "A_norm_bound": native.radial_support_m*norm*axis_norm,
                         "jac_A_norm_bound": norm*axis_norm*rotation_norm})
        return {"source_id": self.source_id, "local_radius_m": native.radial_support_m,
                "scope": "ordinary derivatives on open raster lines and open axial support sides", "derivatives": rows}

    def to_metadata(self):
        return {"schema": "native-dynamic-dipole-v1", "source_id": self.source_id,
                "owner_ids": self.owner_ids, "base_domain": self.base.to_metadata(),
                "drives": [drive.to_dict() for drive in self.drives], "reference_time_s": self.reference_time_s,
                "static": False, "dynamic_drive_frozen": False, "induced_electric_field_modelled": True,
                "time_reference": "absolute laboratory seconds", "temporal_bounds": self.temporal_bounds(),
                "limitations": ("Prescribed native uniform-coil Az; no circuit, fringe or radiation solve",
                                "Raster flyback impulses are not replaced by zero or a frozen magnetic field")}

    def _payload(self):
        return {"base": self.base.to_dict(), "owner_ids": self.owner_ids,
                "drives": [drive.to_dict() for drive in self.drives], "reference_time_s": self.reference_time_s}

    def to_dict(self):
        return {**self._payload(), "digest": self.digest}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, Mapping) or set(record) != {"base", "owner_ids", "drives", "reference_time_s", "digest"}:
            raise ValueError("Invalid dynamic coil record")
        result = cls(AnalyticPotentialDomain.from_dict(record["base"]), tuple(record["owner_ids"]),
                     tuple(NativeTransverseDrive.from_dict(value) for value in record["drives"]), record["reference_time_s"])
        if result.digest != record["digest"]:
            raise ValueError("Dynamic coil identity mismatch")
        return result


@dataclass(frozen=True, slots=True)
class DynamicPotentialSnapshot:
    """One immutable source sum shared by time-resolved particle/wave solvers."""

    static_snapshot: ElectromagneticPotentialSnapshot
    dynamic_coils: tuple[DynamicDipolePotential, ...]
    numerical_digest: str = field(init=False)

    def __post_init__(self):
        if type(self.static_snapshot) is not ElectromagneticPotentialSnapshot:
            raise ValueError("Dynamic potentials require the exact retained static snapshot")
        coils = tuple(self.dynamic_coils)
        if not coils or any(type(coil) is not DynamicDipolePotential for coil in coils):
            raise ValueError("Dynamic snapshot requires exact immutable native coils")
        object.__setattr__(self, "dynamic_coils", coils)
        ids, ownership = set(), set()
        for row in self.contributions:
            if row.source_id in ids:
                raise ValueError(f"Duplicate potential source: {row.source_id}")
            ids.add(row.source_id)
            for owner in row.owner_ids:
                atomic = (row.kind, _canonical_owner(self.scene, owner))
                if atomic in ownership:
                    raise ValueError(f"Duplicate static/dynamic physical field coverage: {owner}")
                ownership.add(atomic)
        object.__setattr__(self, "numerical_digest", json_digest(self._payload()))

    @property
    def scene(self):
        return self.static_snapshot.scene

    @property
    def gauge(self):
        return self.static_snapshot.gauge

    @property
    def contributions(self):
        return (*self.static_snapshot.contributions, *self.dynamic_coils)

    @property
    def physical_digest(self):
        return json_digest({"static": self.static_snapshot.physical_digest,
                            "dynamic_coils": [coil.digest for coil in self.dynamic_coils]})

    def sample(self, positions_m, time_s):
        result = self.static_snapshot.sample(positions_m)
        result.update(dA_dt_t_m_per_s=np.zeros_like(result["A_t_m"]),
                      d2A_dt2_t_m_per_s2=np.zeros_like(result["A_t_m"]),
                      dphi_dt_v_per_s=np.zeros_like(result["phi_v"]))
        for coil in self.dynamic_coils:
            for name, value in coil.sample(positions_m, time_s).items():
                result[name] += value
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Combined dynamic potential exceeds the finite numerical range")
        return result

    def sample_dynamic_jet(self, positions_m, time_s):
        result = self.dynamic_coils[0].sample_jet(positions_m, time_s)
        for coil in self.dynamic_coils[1:]:
            for key, value in coil.sample_jet(positions_m, time_s).items():
                result[key] += value
        return result

    def require_time_interval(self, start_s, stop_s):
        for coil in self.dynamic_coils:
            coil.require_time_interval(start_s, stop_s)

    def next_time_event(self, time_s):
        return min(coil.next_time_event(time_s) for coil in self.dynamic_coils)

    def temporal_bounds(self):
        return tuple(coil.temporal_bounds() for coil in self.dynamic_coils)

    def to_metadata(self):
        return {"schema": SCHEMA, "static": False, "numerical_digest": self.numerical_digest,
                "static_snapshot_digest": self.static_snapshot.numerical_digest,
                "dynamic_coils": [coil.to_metadata() for coil in self.dynamic_coils],
                "electric_policy": "retained -grad(phi) plus exact ordinary -partial_t(A)",
                "time_reference": "absolute laboratory seconds; caller supplies emission epoch plus flight time"}

    def _payload(self):
        return {"schema": SCHEMA, "static_snapshot": self.static_snapshot.to_dict(),
                "dynamic_coils": [coil.to_dict() for coil in self.dynamic_coils]}

    def to_dict(self):
        return {**self._payload(), "numerical_digest": self.numerical_digest}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, Mapping) or set(record) != {"schema", "static_snapshot", "dynamic_coils", "numerical_digest"} or record["schema"] != SCHEMA:
            raise ValueError("Invalid dynamic potential snapshot record")
        result = cls(ElectromagneticPotentialSnapshot.from_dict(record["static_snapshot"]),
                     tuple(DynamicDipolePotential.from_dict(value) for value in record["dynamic_coils"]))
        if result.numerical_digest != record["numerical_digest"]:
            raise ValueError("Dynamic potential snapshot identity mismatch")
        return result


def capture_native_dynamic_coil(row, components):
    """Replace one captured native coil by its complete immutable time law."""
    from temsim.optics.ac_deflector import AcDeflectorComponent
    from temsim.optics.descan_deflector import DescanDeflectorComponent
    if type(row.potential) is not AnalyticMagneticPotential or type(row.potential.field) is not FrozenPosedMultipole:
        raise ValueError("Native dynamic capture requires a retained posed dipole")
    native = row.potential.field
    coil_index = int(native.lens_key.rsplit(":", 1)[1])
    laws = []
    for key in native.drive_keys:
        component = components[key]
        if not (getattr(component, "scan_enabled", False) or getattr(component, "wobble_enabled", False)):
            continue
        if type(component) not in (AcDeflectorComponent, DescanDeflectorComponent):
            raise ValueError(f"{key}: unsupported native dynamic component type")
        methods = ("kick_events", "coil_kick_matrices", "coil_kicks_mrad", "dynamic_kick_mrad",
                   "instantaneous_kick_mrad", "scan_kick_mrad", "scan_factors")
        if type(component) is AcDeflectorComponent:
            methods = (*methods,"wobble_factor")
        for name in methods:
            if getattr(getattr(component, name), "__func__", None) is not getattr(type(component), name):
                raise ValueError(f"{key}: overridden native drive methods cannot be frozen")
        coupling = np.asarray(component.coil_kick_matrices()[coil_index], dtype=float)
        if component.scan_enabled:
            law = NativeTransverseDrive(key, "raster", coupling@np.asarray(component.scan_command_matrix_mrad),
                                        component.scan_frame_period_s, component.scan_lines)
        else:
            amplitude = coupling@np.array((component.wobble_amplitude_x_mrad, component.wobble_amplitude_y_mrad))
            law = NativeTransverseDrive(key, "sine", np.c_[amplitude, np.zeros(2)], component.wobble_period_s,
                                        phase_rad=math.radians(component.wobble_phase_deg))
        if law.varying:
            laws.append(law)
    if not laws:
        return None
    return DynamicDipolePotential(AnalyticPotentialDomain(row.potential), row.owner_ids, tuple(laws), native.captured_time_s)
