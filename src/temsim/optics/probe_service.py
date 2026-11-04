"""Percent-based probe outputs, intrinsic service tools and optional learning.

All percentages refer to the installed model's declared output limits. Local
electrical bases retain their installation frame. Learning is explicit
feed-forward calibration and starts disabled with zero coefficients.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
import math
from numbers import Real

from temsim.runtime_parameters import RuntimeTarget, runtime_targets, validate_runtime_assignment

PERCENT_SCHEMA = "probe-service-percent-v1"


@dataclass(frozen=True)
class Output:
    component: str
    unit: str
    limit_field: str
    fields: tuple[tuple[str, float], ...]
    frame: str


def _outputs():
    result = {}

    def direct(name, key, unit, limit, field, frame):
        result[name] = Output(key, unit, limit, ((field, 1.),), frame)

    for name in ("HP1", "HP2"):
        direct(name, f"probe_{name.lower()}_hexapole", "m⁻³", "maximum_strength_m3",
               "strength_m3", "Scalar main hexapole drive at its fixed installation angle.")
    for name in ("TL12", "TL21", "TL22", "ADL"):
        key = "adapter_lens" if name == "ADL" else f"probe_{name.lower()}_lens"
        direct(name, key, "% excitation", "max_percent", "percent", "Scalar round-lens excitation.")
    for axis, normal in (("x", True), ("y", False)):
        for name in ("DP11", "DP21", "DP22", "DPH1", "DPH2"):
            direct(name + axis, f"probe_{name.lower()}_deflector", "mrad", "maximum_kick_mrad",
                   f"kick_{axis}_mrad", "Local transverse deflection basis at the physical coil host.")
        direct("DP12" + axis, "probe_dp12_scan_deflector", "mrad", "maximum_kick_mrad",
               f"alignment_{axis}_mrad", "Virtual alignment basis realized by the two physical AC coils.")
        for name in ("QPol", "QPC", "QPH1", "QPH2"):
            direct(name + axis, f"probe_{name.lower()}_quadrupole", "m⁻²", "maximum_strength_m2",
                   "strength_m2" if normal else "skew_strength_m2",
                   "Local quadrupole normal/skew field bases, separated by 45 degrees.")
        for name in ("HPol", "HPC"):
            direct(name + axis, f"probe_{name.lower()}_hexapole", "m⁻³", "maximum_strength_m3",
                   "strength_m3" if normal else "skew_strength_m3",
                   "Local hexapole normal/skew field bases, separated by 30 degrees at a fixed mount angle.")
        for name, upper, lower in (("BSh", 1., -1.), ("BTlt", .5, .5)):
            result[name + axis] = Output("beam_deflector", "mrad", "maximum_kick_mrad", (
                (f"upper_{axis}_mrad", upper), (f"lower_{axis}_mrad", lower)),
                "Difference/sum mode of the upstream beam pair in its local deflection frame.")
    return result


OUTPUTS = _outputs()


@dataclass(frozen=True)
class ServiceRoute:
    tool: str
    input_name: str
    intrinsic: tuple[tuple[str, float | None], ...]
    learning: tuple[str, ...]
    evidence: str

    @property
    def key(self):
        return f"{self.tool}.{self.input_name}"


def _routes():
    beam = ("BShx", "BShy", "BTltx", "BTlty")
    relay = ("DP21x", "DP21y", "DP22x", "DP22y")
    local = ("DPH1x", "DPH1y", "DPH2x", "DPH2y", "QPH1x", "QPH1y", "QPH2x", "QPH2y")
    unknown = "This input's intrinsic weights need calibration. Blank weights hold those outputs fixed."
    rows = [
        ServiceRoute("TL2SymFoc", "single",
                     tuple((n, None) for n in ("TL12", "TL21", "TL22", "ADL", *relay)), (),
                     "Note 7 identifies the focus-related composite tool. " + unknown),
        ServiceRoute("AT_A1", "upper", (("DP21x", 1.), ("DP22x", 1.)) + tuple((n, None) for n in beam), (),
                     "Note 10 describes a symmetric DP pair and beam controls. Equal normalized DP weights are the model basis."),
        ServiceRoute("AT_A1", "lower", tuple((n, None) for n in ("DP21y", "DP22y", *beam)), (), unknown),
        ServiceRoute("AT_A1fine", "upper", (("QPolx", 1.),), ("DP11x", "DP11y"),
                     "Note 11. QPol is intrinsic. The small dipole changes belong to learning."),
        ServiceRoute("AT_A1fine", "lower", (("QPoly", None),), ("DP11x", "DP11y"), unknown),
        ServiceRoute("AT_A2", "upper", (("HP1", 1.), ("HP2", -1.)),
                     ("DPH1x", "DPH1y", "DPH2x", "DPH2y", "QPH1x"),
                     "Main hexapole difference. Auxiliary small changes belong to learning."),
        ServiceRoute("AT_A2", "lower", (("TL21", 1.), ("TL22", -1.)), ("DP22x", "DP22y"),
                     "Transfer-lens difference. The small DP22 response belongs to learning."),
        ServiceRoute("AT_A2fine", "upper", (("HPolx", 1.),), ("QPoly",),
                     "HPolx is intrinsic. Small QPoly cross-compensation belongs to learning."),
        ServiceRoute("AT_A2fine", "lower", (("HPoly", 1.),), ("QPolx",),
                     "HPoly is intrinsic. Small QPolx cross-compensation belongs to learning."),
        ServiceRoute("AT_C3", "single", (("HP1", 1.), ("HP2", 1.)), local,
                     "Model common main-hexapole drive. The UI says AT_C3 and note 22 says AT_C2. Auxiliary trims are learning."),
        ServiceRoute("AT_A3C", "upper", (("QPCx", 1.),), relay,
                     "QPCx is intrinsic. Small DP adjustments belong to learning."),
        ServiceRoute("AT_A3C", "lower", (("QPCy", None),), relay, unknown),
    ]
    for selector, name, gain in (("upper", "QPolx", 1.), ("lower", "QPoly", None)):
        rows.append(ServiceRoute("AT_S3", selector, ((name, gain),),
                    ("DP11x", "DP11y", "DP12x", "DP12y", *relay, *beam),
                    "QPol is intrinsic. The note does not distinguish both inputs. "
                    "Small steering changes belong to learning. A1 and S3 share a primary field and require response calibration."))
    for tool in ("AT_B2", "AT_B2fine"):
        for axis in ("x", "y"):
            rows.append(ServiceRoute(tool, axis, (("DP12" + axis, 1.),), (),
                        "Both tools use the same intrinsic DP12 output on the AC coil pair."))
    for tool, sign in (("TL2sym", 1.), ("TL2asym", -1.)):
        rows.append(ServiceRoute(tool, "single", (("TL21", 1.), ("TL22", sign)), (),
                                 "Common/differential normalized transfer-lens excitation."))
    for tool, sign in (("DP2sym", 1.), ("DP2asym", -1.)):
        for axis in ("x", "y"):
            rows.append(ServiceRoute(tool, axis, (("DP21" + axis, 1.), ("DP22" + axis, sign)), (),
                                     "Common/differential normalized deflection."))
    for stage in ("1", "2"):
        learning = (f"DPH{stage}x", f"DPH{stage}y")
        rows.append(ServiceRoute(f"HP{stage}_P", "single", ((f"HP{stage}", 1.),), learning,
                                 "The main hexapole is intrinsic. Local dipole compensation belongs to learning."))
        for axis in ("x", "y"):
            rows.append(ServiceRoute(f"QPH{stage}_P", axis, ((f"QPH{stage}{axis}", 1.),), learning,
                                     "The named quadrupole output is intrinsic. Small dipole compensation belongs to learning."))
    return {row.key: row for row in rows}


ROUTES = _routes()


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite and numeric")
    return float(value)


def validated_service_settings(settings):
    """Retain historical count gains as inert metadata, preserving their values."""
    if not isinstance(settings, Mapping):
        raise ValueError("Legacy service gains must be a mapping")
    result = {}
    for key, values in settings.items():
        if key not in ROUTES or not isinstance(values, Mapping):
            raise ValueError("Unknown legacy service input")
        result[key] = {}
        for name, value in values.items():
            if name not in OUTPUTS:
                raise ValueError(f"Unknown legacy output {name}")
            result[key][name] = _number(value, name)
    return result


def validated_percent_settings(settings):
    if not isinstance(settings, Mapping) or set(settings) - {"schema", "learning_enabled", "intrinsic", "learning"}:
        raise ValueError("Invalid percent service settings")
    if settings.get("schema", PERCENT_SCHEMA) != PERCENT_SCHEMA:
        raise ValueError("Unknown percent service schema")
    enabled = settings.get("learning_enabled", False)
    if type(enabled) is not bool:
        raise ValueError("Learning selection must be Boolean")
    result = dict(schema=PERCENT_SCHEMA, learning_enabled=enabled, intrinsic={}, learning={})
    for section in ("intrinsic", "learning"):
        table = settings.get(section, {})
        if not isinstance(table, Mapping):
            raise ValueError(f"{section} must be a service matrix")
        for key, row in table.items():
            if key not in ROUTES or not isinstance(row, Mapping):
                raise ValueError("Unknown service input")
            allowed = dict(ROUTES[key].intrinsic) if section == "intrinsic" else ROUTES[key].learning
            if set(row) - set(allowed):
                raise ValueError(f"Output lies outside the {section} group for {key}")
            result[section][key] = {n: _number(v, n) for n, v in row.items()}
    return result


def service_settings(state, key):
    if key not in ROUTES:
        raise ValueError(f"Unknown service input {key}")
    saved = validated_percent_settings(getattr(state, "probe_service_percent", {}))
    primary = {n: 0. if v is None else v for n, v in ROUTES[key].intrinsic}
    primary.update(saved["intrinsic"].get(key, {}))
    learning = dict.fromkeys(ROUTES[key].learning, 0.)
    learning.update(saved["learning"].get(key, {}))
    return primary, learning, saved["learning_enabled"]


@dataclass(frozen=True)
class OutputReading:
    maximum: float
    native: float
    percent: float
    unit: str
    frame: str


def _reading(targets, name):
    if name not in OUTPUTS:
        raise ValueError(f"Unknown probe output {name}")
    output = OUTPUTS[name]
    target = targets.get(output.component)
    if target is None or not getattr(target.obj, "_layout_installed", True):
        raise ValueError(f"{output.component} is not installed")
    item = target.obj
    limit = _number(getattr(item, output.limit_field), "Model maximum")
    if limit <= 0:
        raise ValueError("Model maximum must be positive")
    limit /= max(abs(share) for _, share in output.fields)
    native = sum(getattr(item, field)*share for field, share in output.fields) / sum(
        share*share for _, share in output.fields)
    _number(native, "Current output")
    return OutputReading(limit, native, 100*native/limit, output.unit, output.frame)


def output_reading(state, name):
    if not getattr(state, "probe_corrector_installed", False):
        raise ValueError("Install the probe corrector first")
    return _reading(runtime_targets(state), name)


@dataclass(frozen=True)
class DriveChange:
    component: str
    field: str
    unit: str
    before: float
    after: float


def _prepare_outputs(state, increments, *, native=False):
    if not getattr(state, "probe_corrector_installed", False):
        raise ValueError("Install the probe corrector first")
    targets = runtime_targets(state)
    deltas = defaultdict(float)
    for name, increment in increments.items():
        increment = _number(increment, "Output increment")
        output = OUTPUTS[name]
        maximum = _reading(targets, name).maximum
        delta = increment if native else maximum * increment / 100.
        for field, share in output.fields:
            deltas[output.component, field, output.unit] += delta*share
    staged, changes = {}, []
    for (key, field, unit), delta in deltas.items():
        if not delta:
            continue
        target = targets[key]
        if not getattr(target.obj, "enabled", False):
            raise ValueError(f"{key} must be enabled")
        if key not in staged:
            staged[key] = RuntimeTarget(key, target.label, copy(target.obj))
        before = getattr(target.obj, field)
        after = before + delta
        if not math.isfinite(after) or after == before:
            raise ValueError(f"{key}.{field} exceeds numerical range or resolution")
        setattr(staged[key].obj, field, after)
        changes.append(DriveChange(key, field, unit, before, after))
    if not changes:
        raise ValueError("The selected adjustment has no nonzero output")
    for change in changes:
        validate_runtime_assignment(staged[change.component], change.field, change.after)
    for target in staged.values():
        validate = getattr(target.obj, "validate", None)
        if validate:
            validate()
    if "probe_dp12_scan_deflector" in staged:
        ac = targets.get("ac_deflector")
        if ac is None or not ac.obj.enabled or not getattr(ac.obj, "_layout_installed", True):
            raise ValueError("DP12 requires the installed, enabled AC scan coil pair")
        ac.obj.validate_coil_limits(staged["probe_dp12_scan_deflector"].obj.coil_bias_mrad(ac.obj))
    return targets, tuple(changes)


def _service_move(state, key, delta_percent, intrinsic, learning, learning_enabled):
    delta = _number(delta_percent, "Service increment (%)")
    if not delta or abs(delta) > 100:
        raise ValueError("Service increment must be nonzero and within ±100%")
    defaults, corrections, enabled = service_settings(state, key)
    saved = validated_percent_settings(getattr(state, "probe_service_percent", {}))
    if intrinsic is not None and not isinstance(intrinsic, Mapping):
        raise ValueError("Intrinsic weights must be a mapping")
    if learning is not None and not isinstance(learning, Mapping):
        raise ValueError("Learning weights must be a mapping")
    saved["intrinsic"][key] = {**defaults, **(intrinsic or {})}
    saved["learning"][key] = {**corrections, **(learning or {})}
    saved["learning_enabled"] = enabled if learning_enabled is None else learning_enabled
    saved = validated_percent_settings(saved)
    weights = dict(saved["intrinsic"][key])
    if saved["learning_enabled"]:
        for name, gain in saved["learning"][key].items():
            weights[name] = weights.get(name, 0.) + gain
    targets, changes = _prepare_outputs(state, {name: delta*gain for name, gain in weights.items() if gain})
    return targets, changes, saved


def preview_probe_service(state, key, *, delta_percent, intrinsic=None, learning=None, learning_enabled=None):
    return _service_move(state, key, delta_percent, intrinsic, learning, learning_enabled)[1]


def _commit(targets, changes, expected):
    if expected is not None and changes != expected:
        raise ValueError("The displayed drives changed. Review the fresh preview.")
    applied = []
    try:
        for change in changes:
            applied.append(change)
            setattr(targets[change.component].obj, change.field, change.after)
    except BaseException:
        for change in reversed(applied):
            setattr(targets[change.component].obj, change.field, change.before)
        raise


def apply_probe_service(state, key, *, delta_percent, intrinsic=None, learning=None, learning_enabled=None, expected=None):
    targets, changes, saved = _service_move(state, key, delta_percent, intrinsic, learning, learning_enabled)
    _commit(targets, changes, expected)
    state.probe_service_percent = saved
    return changes


def preview_probe_output(state, name, *, percent):
    value = _number(percent, "Output setting (%)")
    if abs(value) > 100:
        raise ValueError("Output setting must be within ±100%")
    current = output_reading(state, name)
    target = current.maximum * value / 100.
    if math.isclose(current.native, target, rel_tol=0., abs_tol=current.maximum*1e-16):
        return ()
    return _prepare_outputs(state, {name: target-current.native}, native=True)[1]


def set_probe_output_percent(state, name, *, percent, expected=None):
    value = _number(percent, "Output setting (%)")
    if abs(value) > 100:
        raise ValueError("Output setting must be within ±100%")
    current = output_reading(state, name)
    target = current.maximum * value / 100.
    if math.isclose(current.native, target, rel_tol=0., abs_tol=current.maximum*1e-16):
        if expected:
            raise ValueError("The displayed drives changed. Review the fresh preview.")
        return ()
    targets, changes = _prepare_outputs(state, {name: target-current.native}, native=True)
    _commit(targets, changes, expected)
    return changes
