"""Detached assembly-error, aberration and electrical-headroom experiments.

The nominal captured instrument is the reference, not an assumed perfect OEM
instrument. Service-route membership defines intrinsic versus learning outputs;
natural cross-aberration response is never relabelled as learning. Only the
explicitly saved learning matrix is used. This is a fixed-pupil diagnostic,
not a finite-source transmission or manufacturing qualification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Real
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim import input_io
from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot
from temsim.lens_pose import physical_pose_values, pose_control_values, supports_physical_column_pose
from temsim.optics.aberration_basis import CARTESIAN_NAMES, cartesian_coefficients
from temsim.optics.probe_service import ROUTES, service_settings
from temsim.optics.system_residual_aberrations import (
    capture_system_residual_reference,
)

SCHEMA = "assembly-error-study-v1"
ERROR_FIELDS = ("offset_x_mm", "offset_y_mm", "offset_z_mm", "rotation_x_mrad",
                "rotation_y_mrad", "assembly_azimuth_deg", "field_azimuth_error_deg")
THIRD_ORDER_NAMES = CARTESIAN_NAMES[:12]
SCOPE = (
    "Probe-column fixed-pupil diagnostic relative to the captured nominal assembly. "
    "All cases use the same input rays, nominal transfer, energy and observation plane. "
    "Only selected service-route electrical outputs may compensate errors. "
    "Natural physical coupling remains active with learning off. Learning means the "
    "separately saved feed-forward gains, not observed cross-aberration response. "
    "No OEM gains, automatic learning fit, lens refocus, source renormalization or "
    "aberration-screen correction is introduced. Finite-source current, clipping, "
    "sampling/step convergence and physical qualification are NOT_ASSESSED."
)


def _finite(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite and numeric")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


@dataclass(frozen=True)
class AssemblyError:
    component: str
    parameter: str
    delta: float

    def __post_init__(self):
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("Specify a component key")
        if self.parameter not in ERROR_FIELDS:
            raise ValueError("Select a supported column offset, optical tilt or azimuth")
        object.__setattr__(self, "delta", _finite(self.delta, "Assembly error"))


@dataclass(frozen=True)
class AssemblyErrorCase:
    name: str
    errors: tuple[AssemblyError, ...]

    def __post_init__(self):
        object.__setattr__(self, "errors", tuple(self.errors))
        if not isinstance(self.name, str) or not self.name.strip() or not 1 <= len(self.errors) <= 32:
            raise ValueError("A named case requires 1 to 32 assembly errors")
        if any(not isinstance(e, AssemblyError) for e in self.errors):
            raise ValueError("Invalid assembly error")
        if len({(e.component, e.parameter) for e in self.errors}) != len(self.errors):
            raise ValueError("Combine duplicate component/parameter errors into one delta")


@dataclass(frozen=True)
class AssemblyStudyOptions:
    routes: tuple[str, ...] = ("AT_A2fine.upper", "AT_A2fine.lower")
    target_coefficients: tuple[str, ...] = THIRD_ORDER_NAMES
    compare_learning: bool = False
    semiangle_mrad: float = .5
    maximum_step_mm: float = .05
    calibration_step_percent: float = .01
    maximum_change_percent: float = 1.
    maximum_iterations: int = 3
    pupil_rms_tolerance_nm: float = .01
    centroid_tolerance_nm: float = .1
    direction_tolerance_mrad: float = .001

    def __post_init__(self):
        for name in ("routes", "target_coefficients"):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"Select distinct {name}")
        if len(self.routes) > 8 or any(key not in ROUTES for key in self.routes):
            raise ValueError("Select 1 to 8 registered probe service inputs")
        if any(key not in THIRD_ORDER_NAMES for key in self.target_coefficients):
            raise ValueError("Targets must be measured Cartesian terms through third order")
        if type(self.compare_learning) is not bool:
            raise ValueError("Learning comparison must be explicitly Boolean")
        if type(self.maximum_iterations) is not int or not 1 <= self.maximum_iterations <= 12:
            raise ValueError("Use 1 to 12 local correction iterations")
        for name in ("semiangle_mrad", "maximum_step_mm", "calibration_step_percent",
                     "maximum_change_percent", "pupil_rms_tolerance_nm", "centroid_tolerance_nm",
                     "direction_tolerance_mrad"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))
        if self.semiangle_mrad > 100 or self.maximum_change_percent > 100:
            raise ValueError("Pupil and route search bounds must not exceed 100")
        if self.calibration_step_percent > self.maximum_change_percent:
            raise ValueError("Calibration step must fit inside the declared route search bound")


@dataclass(frozen=True)
class AssemblyStudyRequest:
    snapshot: InstrumentSnapshot
    cases: tuple[AssemblyErrorCase, ...]
    options: AssemblyStudyOptions = field(default_factory=AssemblyStudyOptions)

    def __post_init__(self):
        object.__setattr__(self, "cases", tuple(self.cases))
        if not isinstance(self.snapshot, InstrumentSnapshot) or not isinstance(self.options, AssemblyStudyOptions):
            raise ValueError("Study requires a complete working point and validated options")
        if not 1 <= len(self.cases) <= 16 or any(not isinstance(c, AssemblyErrorCase) for c in self.cases):
            raise ValueError("Provide 1 to 16 assembly-error cases")
        if len({c.name for c in self.cases}) != len(self.cases):
            raise ValueError("Case names must be unique")

    def to_dict(self):
        return dict(schema=SCHEMA, snapshot=self.snapshot.to_dict(),
                    cases=[asdict(c) for c in self.cases], options=asdict(self.options))

    @property
    def digest(self):
        return json_digest(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"schema", "snapshot", "cases", "options"} or data["schema"] != SCHEMA:
            raise ValueError("Unsupported assembly study request")
        return cls(InstrumentSnapshot.from_dict(data["snapshot"]),
                   tuple(AssemblyErrorCase(c["name"], tuple(AssemblyError(**e) for e in c["errors"]))
                         for c in data["cases"]), AssemblyStudyOptions(**data["options"]))


def supported_error_parts(state):
    """Only actual supported placement owners upstream of the probe plane."""
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is None:
        return ()
    return tuple(p for p in assembly.parts if p.module_key.startswith("column:")
                 and p.center_z_mm < state.sample.z_mm and supports_physical_column_pose(p.data))


def capture_study(state, cases, options=AssemblyStudyOptions()):
    """Capture all inputs, including portable TOMLs; never alter the live state."""
    snapshot = capture_instrument_snapshot(state)
    work = snapshot.restore()
    if not getattr(work, "probe_corrector_installed", False):
        raise ValueError("Assembly correction studies require an installed probe corrector")
    input_io.bind_archive(work, input_io.capture_input_archive(work))
    request = AssemblyStudyRequest(capture_instrument_snapshot(work), tuple(cases), options)
    _route_weights(work, options)
    for case in request.cases:
        _geometry_changes(work, case)
    return request


def _geometry_changes(state, case):
    parts = {part.key: part for part in supported_error_parts(state)}
    changes, evidence = [], []
    for error in case.errors:
        part = parts.get(error.component)
        if part is None or error.parameter not in pose_control_values(part.data):
            raise ValueError(f"{error.component}: select a supported upstream physical placement owner")
        before = physical_pose_values(part.data)[error.parameter]
        after = _finite(before + error.delta, "Perturbed placement")
        changes.append(("column", error.component, error.parameter, after))
        evidence.append(dict(**asdict(error), before=before, after=after,
                             unit=error.parameter.rsplit("_", 1)[-1]))
    return changes, evidence


def prepare_error_case(request, case, *, cancelled=lambda: False):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.geometry_experiments import prepare_geometry_changes
    work = request.snapshot.restore()
    changes, errors = _geometry_changes(work, case)
    with input_io.input_scope(work):
        catalog = AssemblyCatalog(root=work._resolved_assembly.root)
        selection = catalog.selection_for_resolved(work._resolved_assembly)
    work, geometry = prepare_geometry_changes(work, selection, changes,
        input_io.capture_input_archive(work), cancelled=cancelled)
    return work, dict(errors=errors, geometry=geometry)


def _route_weights(state, options):
    intrinsic, learning, evidence = [], [], []
    for key in options.routes:
        primary, auxiliary, saved_enabled = service_settings(state, key)
        primary = {n: v for n, v in primary.items() if v}
        auxiliary = {n: v for n, v in auxiliary.items() if v}
        if not primary:
            raise ValueError(f"{key}: intrinsic weights are uncalibrated or zero; supply explicit service gains first")
        intrinsic.append(primary)
        learning.append(auxiliary)
        evidence.append(dict(route=key, intrinsic=primary, learning=auxiliary,
            allowed_learning_outputs=ROUTES[key].learning, saved_learning_enabled=saved_enabled,
            interpretation="Service inputs, not laboratory X/Y aberration knobs",
            source=ROUTES[key].evidence))
    return intrinsic, learning, evidence


def report_vector(report):
    d = report.diagnostics
    vector = np.r_[cartesian_coefficients(report.coefficients),
                   d["reference_centroid_xy_m"], d["reference_centroid_slopes_rad"]]
    if vector.shape != (19,) or not np.isfinite(vector).all():
        raise ValueError("Finite coefficients, chief position and mechanical direction are required")
    return vector


def report_payload(report):
    report_vector(report)
    return dict(coefficients_mm=dict(zip(CARTESIAN_NAMES, map(float, cartesian_coefficients(report.coefficients)))),
                diagnostics=thaw_json(freeze_json(report.diagnostics)))


def run_study(request, *, cancelled=lambda: False, progress=lambda message: None):
    """Compare every case to one frozen nominal reference; retain failed cases."""
    from temsim.assembly_error_response import StudyObserver, compensate, response_matrix, combined_weights
    started = perf_counter()
    if cancelled():
        raise RuntimeError("Assembly study cancelled")
    nominal = request.snapshot.restore()
    if not getattr(nominal, "probe_corrector_installed", False):
        raise ValueError("Assembly correction studies require an installed probe corrector")
    intrinsic, learning, routes = _route_weights(nominal, request.options)
    progress("Capturing the nominal pupil and aberrations")
    reference = capture_system_residual_reference(nominal, "probe",
        semiangle_mrad=request.options.semiangle_mrad,
        maximum_step_mm=request.options.maximum_step_mm, cancelled=cancelled)
    observer = StudyObserver(reference, request.options, cancelled, progress)
    observer.validate_state(nominal)
    baseline = observer.measure(nominal)
    rows = []
    for case in request.cases:
        if cancelled():
            raise RuntimeError("Assembly study cancelled")
        progress(f"{case.name}: applying archived assembly errors")
        row = dict(name=case.name, errors=[asdict(e) for e in case.errors])
        try:
            work, evidence = prepare_error_case(request, case, cancelled=cancelled)
            row["assembly"] = evidence
            observer.validate_state(work)
            perturbed = observer.measure(work)
            row["uncompensated"] = report_payload(perturbed)
            row["uncompensated_errors"] = observer.errors(perturbed, baseline)
            row["intrinsic_only"] = compensate(work, baseline, perturbed, intrinsic, observer)
            row["status"] = "COMPLETE"
            if request.options.compare_learning:
                if any(learning):
                    try:
                        progress(f"{case.name}: measuring the separately configured learning response")
                        row["learning_only_response"] = response_matrix(work, perturbed, learning, observer)[1]
                        row["with_learning"] = compensate(work, baseline, perturbed,
                            combined_weights(intrinsic, learning), observer)
                    except (ValueError, RuntimeError) as exc:
                        if cancelled():
                            raise
                        row["with_learning"] = dict(status="FAILED", failure=str(exc))
                        row["status"] = "PARTIAL"
                else:
                    row["with_learning"] = dict(row["intrinsic_only"],
                        reused_evidence="Identical electrical commands: all selected learning gains are zero")
            else:
                row["with_learning"] = dict(status="NOT_REQUESTED")
        except (ValueError, RuntimeError) as exc:
            if cancelled():
                raise RuntimeError("Assembly study cancelled") from exc
            row.update(status="FAILED", failure=str(exc))
        rows.append(row)
    if cancelled():
        raise RuntimeError("Assembly study cancelled")
    return freeze_json(dict(schema=SCHEMA, request_digest=request.digest, scope=SCOPE,
        options=asdict(request.options),
        nominal=report_payload(baseline), routes=routes, cases=rows,
        reference=dict(system="probe", energy_kev=reference.energy_kev,
                       z0_mm=reference.z0_mm, z1_mm=reference.z1_mm,
                       pupil_reference=reference.pupil_reference,
                       matrix=reference.matrix.tolist(), inputs=reference.inputs.tolist()),
        measurement_count=observer.count, elapsed_s=perf_counter()-started,
        finite_source=dict(status="NOT_ASSESSED", emitted=None, surviving=None, effective=None,
                           current_a=None, reason="Zero-current diagnostic pupil; no finite-source calculation"),
        qualification="NOT_QUALIFIED; local diagnostic targets and model limits only"))


def save_study(path, request, result):
    """Save a checksum-bound portable request and its detached report atomically."""
    import os
    import tempfile
    if result["request_digest"] != request.digest:
        raise ValueError("Study result does not belong to these captured inputs")
    payload = dict(schema=SCHEMA, request=request.to_dict(), result=thaw_json(result))
    document = dict(payload, checksum=json_digest(payload))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_study(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    checksum = data.pop("checksum", None)
    if data.get("schema") != SCHEMA or checksum != json_digest(data):
        raise ValueError("Assembly study checksum/schema mismatch")
    request = AssemblyStudyRequest.from_dict(data["request"])
    if data["result"]["request_digest"] != request.digest:
        raise ValueError("Study report belongs to different captured inputs")
    return request, freeze_json(data["result"])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Saved assembly study JSON")
    parser.add_argument("--output", type=Path, required=True, help="Destination for a newly executed report")
    args = parser.parse_args()
    if args.study.resolve() == args.output.resolve():
        parser.error("Keep the original study; choose a different output path")
    request, _ = load_study(args.study)
    result = run_study(request, progress=print)
    save_study(args.output, request, result)
    print(f"Saved {len(result['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
