"""Collect physical conditional angular-CDF boundary probes for Image optics.

The five other source CDF coordinates are fixed scrambled Sobol points. Every
radial-CDF query is emitted at the physical tip with explicit +/- source signs,
and traced through a fresh gun before four C2 cases reuse that same gun. Probe
weights are exactly zero: these traces locate a boundary and predict no dose.

Globally batched roots partition the full radial CDF. Positive-weight Gauss
sources are then emitted through a fresh physical gun and the actual screen.
Finite scanning, root tolerance and propagation error are reported separately.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "temsim").is_dir())
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.validate_image_intensity_convergence import (
    CASES, array_digest, canonical_digest, digest_file, screen_arrays, source_arrays, write_json, write_npz,
)

SCHEMA = "conditional-source-convergence-v1"
CHANNELS = tuple(f"{case}:{sign:+d}" for case in CASES for sign in (1, -1))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=Path("outputs/c2-screen-aberrations-20261017"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/image-conditional-source-pilot-20261017"))
    parser.add_argument("--base-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--azimuth-orbit-size", type=int, choices=(2,4,8,16), default=2)
    parser.add_argument("--energy-band", type=float, nargs=2, metavar=("LOWER_EV", "UPPER_EV"))
    parser.add_argument("--gun-step-scale", type=float, default=1.)
    parser.add_argument("--step", type=float, default=.00625)
    parser.add_argument("--scan-points", type=int, default=8)
    parser.add_argument("--stage", choices=("prepare", "coarse", "full"), default="prepare")
    parser.add_argument("--scan-intervals", type=int, default=8)
    parser.add_argument("--scan-refinements", type=int, default=1)
    parser.add_argument("--u-tol", type=float, default=1e-10)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--orders", type=int, nargs="+", choices=(2,4), default=[2])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend", choices=("CPU", "Numba CPU", "Prefer GPU", "Require GPU"),
                        default="Require GPU")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.base_count < 4 or args.base_count & (args.base_count-1):
        parser.error("The five-dimensional Sobol base count must be a complete power of two >= 4")
    if not 0 <= args.seed < 2**32 or not 0 < args.step <= .1 or args.threads < 1:
        parser.error("Use an unsigned 32-bit seed, positive step <= .1 mm, and positive thread count")
    if args.scan_points < 3:
        parser.error("At least three radial-CDF scan points are required")
    if (args.scan_intervals < 2 or not 0 <= args.scan_refinements <= 4
            or not 1e-14 <= args.u_tol < .01 or args.maxiter < 1):
        parser.error("Invalid conditional root solver parameters")
    import math
    if not math.isfinite(args.gun_step_scale) or args.gun_step_scale<=0:
        parser.error("Gun step scale must be positive and finite")
    if args.energy_band is not None:
        if not all(math.isfinite(x) for x in args.energy_band) or not 0 < args.energy_band[0] < args.energy_band[1]:
            parser.error("Energy strata need explicit positive finite LOWER_EV < UPPER_EV")
    expanded=args.base_count*(args.azimuth_orbit_size//2)*(3 if args.energy_band is not None else 1)
    radial_nodes=(args.scan_points if args.stage=="coarse" else args.scan_intervals*2**args.scan_refinements+1)
    if 2*expanded*radial_nodes>1048576:
        parser.error("Initial actual source table exceeds the explicit source limit of 1048576 rays")
    return args


def validate_resume_metadata(expected,saved):
    for key,value in expected.items():
        if key not in ("status","batches","runs","sources") and canonical_digest(saved.get(key))!=canonical_digest(value):
            raise ValueError(f"Retained conditional metadata differs from the requested protocol: {key}")


def group_label(base_count, orbit_size, order, energy_stratified=False):
    if energy_stratified:
        return f"M{base_count}_E3_K{orbit_size}_G{order}"
    return f"M{base_count}_G{order}" if orbit_size==2 else f"M{base_count}_K{orbit_size}_G{order}"


def apply_gun_step_scale(gun, scale):
    """Scale only captured gun integration lengths; preserve default exactly."""
    import numpy as np
    import math
    if isinstance(scale,(bool,np.bool_)) or not math.isfinite(float(scale)) or float(scale)<=0:
        raise ValueError("Gun step scale must be positive and finite")
    before={key:float(getattr(gun,key)) for key in ("trace_step_mm","drift_step_mm")}
    after={key:value*float(scale) for key,value in before.items()}
    if any(not math.isfinite(value) or value<=0 for value in (*before.values(),*after.values())):
        raise ValueError("Captured and scaled gun integration lengths must be positive and finite")
    if float(scale)!=1.:
        for key,value in after.items():
            setattr(gun,key,value)
    return dict(scale=float(scale),captured=before,applied=after,
        scope="Gun trace/drift integration lengths only; downstream column step is independent")


def compose_energy_orbit(emitter, original_base, orbit_size, band_ev):
    import numpy as np
    from scripts.source_azimuth_orbit import expand_common_azimuth
    from scripts.source_energy_strata import expand_energy_strata

    strata = expand_energy_strata(emitter, original_base, band_ev)
    orbit = expand_common_azimuth(strata.base_points, orbit_size)
    row = orbit.independent_base_index
    half = orbit_size // 2
    mapping = dict(
        base_probability=strata.base_probability[row] / half,
        independent_base_index=strata.independent_base_index[row],
        energy_stratum_index=strata.stratum_index[row],
        half_orbit_index=orbit.half_orbit_index,
        interval_lower=strata.interval_lower[row],
        interval_upper=strata.interval_upper[row],
    )
    probability = mapping["base_probability"]
    if (np.any(probability <= 0.) or not np.isfinite(probability).all()
            or not np.isclose(probability.sum(), 1., rtol=0., atol=1e-14)):
        raise ValueError("Energy/orbit composition lost the complete source probability")
    by_parent = np.bincount(mapping["independent_base_index"], weights=probability,
                            minlength=len(original_base))
    if not np.allclose(by_parent, 1. / len(original_base), rtol=0., atol=1e-15):
        raise ValueError("Energy/orbit composition changed an original base probability")
    cuts = dict(edges=strata.cuts.edges,
                interval_probability=strata.cuts.interval_probability,
                conditional_minimum_ev=strata.cuts.conditional_minimum_ev,
                conditional_maximum_ev=strata.cuts.conditional_maximum_ev)
    orbit_metadata = dict(orbit.metadata,
        independent_sobol_base_count=len(original_base),
        input_positive_energy_stratum_rows=len(strata.base_points),
        expanded_radial_base_count=len(orbit.base_points),
        source_orientation_count_before_radial_nodes=len(strata.base_points)*orbit_size,
        weighting="Original conditional energy probability / M / (K/2) per radial base, then full radial CDF and +/- each 1/2; no normalization.")
    return dict(points=orbit.base_points, mapping=mapping, cuts=cuts,
                energy_metadata=strata.metadata,
                cuts_metadata=strata.cuts.metadata,
                orbit_metadata=orbit_metadata)


def reweight_radial_table(table, base_probability):
    """Replace the helper's uniform 1/R by original p_base, before +/- signs."""
    import numpy as np
    probability = np.asarray(base_probability, dtype=np.float64)
    if (probability.ndim != 1 or not len(probability)
            or not np.isfinite(probability).all() or np.any(probability <= 0.)
            or not np.isclose(probability.sum(), 1., rtol=0., atol=1e-14)):
        raise ValueError("Expected positive full-support radial-base probabilities")
    ids = np.asarray(table.base_index)
    if np.any(ids < 0) or np.any(ids >= len(probability)):
        raise ValueError("A radial node has no original source-base probability")
    weights = np.asarray(table.weight) * (len(probability) * probability[ids])
    if not np.isclose(weights.sum(), 1., rtol=0., atol=2e-14):
        raise ValueError("The weighted full radial CDF measure does not sum to one")
    return replace(table, weight=weights, diagnostics=dict(table.diagnostics,
        base_measure="Original conditional energy CDF probabilities; uniform helper weights multiplied by R*p_base without normalization"))


def weighted_root_budget(roots, base_probability, channels):
    """Located-root bracket budget; no finite-scan or transport certification."""
    import numpy as np
    probability = np.asarray(base_probability, dtype=np.float64)
    per_channel = {key: 0. for key in channels}
    for root in roots:
        b = int(root["base_index"])
        width = float(root["bracket_width"])
        if not 0 <= b < len(probability) or not np.isfinite(width) or width < 0.:
            raise ValueError("Invalid located-root probability or bracket width")
        per_channel[root["channel_key"]] += probability[b] * width / 2.
    cases = list(dict.fromkeys(key.rsplit(":", 1)[0] for key in channels))
    per_case = {case: sum(value for key, value in per_channel.items()
                         if key.rsplit(":", 1)[0] in ("baseline", case))
                for case in cases if case != "baseline"}
    return dict(per_channel=per_channel, difference_bound_by_case=per_case,
                all_channels_sum=float(sum(per_channel.values())),
                scope="Only located-root brackets under original CDF probabilities; finite scan and unbounded physical transport errors remain separate.")


def sampled_energy_containment(anchor_margins, mapping, channels):
    """Inspect actual u=1 anchors; this is sampled evidence, never certification."""
    import numpy as np
    strata = np.asarray(mapping["energy_stratum_index"], dtype=np.int64)
    margins = np.asarray(anchor_margins, dtype=np.float64)
    if margins.shape != (2 * len(strata), len(channels)) or not np.isfinite(margins).all():
        raise ValueError("Energy containment needs every actual u=0/1, case/sign anchor")
    upper = margins.reshape(len(strata), 2, len(channels))[:, 1, :]
    failed = []
    for b, c in zip(*np.nonzero(((strata[:, None] == 0) & (upper < 0.))
                               | ((strata[:, None] == 2) & (upper >= 0.)))):
        failed.append(dict(expanded_base_index=int(b),
                           independent_base_index=int(mapping["independent_base_index"][b]),
                           energy_stratum_index=int(strata[b]), channel=channels[c],
                           margin_mm=float(upper[b, c])))
    return dict(status="PASS_SAMPLED" if not failed else "FAIL_SAMPLED",
                low_energy_radial_bases=int(np.count_nonzero(strata == 0)),
                high_energy_radial_bases=int(np.count_nonzero(strata == 2)),
                failure_count=len(failed), failures=failed,
                scope="Actual angular CDF=1 anchors: low-energy rows pass every C2 case/sign; high-energy rows stop. This finite check does not certify the entire conditional source support.")


def validate_weighted_case_anchors(tracer, solution, case_anchors, case_index, channels):
    """Gate this actual weighted C2 case before starting its screen transport."""
    if tracer.anchor_reference is None:
        raise ValueError("Weighted source needs the actual boundary-probe anchor reference")
    c0=2*case_index
    report=sampled_energy_containment(case_anchors,tracer.base_plan["mapping"],channels[c0:c0+2])
    if report["failure_count"]:
        raise ValueError("Fresh weighted source endpoint anchors escaped the declared energy band")
    observed=tracer.anchor_reference.copy()
    observed[:,c0:c0+2]=case_anchors
    # This updates independent per-channel maxima. Other channels retain their
    # measured reference; this array does not purport to be a whole new batch.
    tracer.observe_anchors(observed)
    drift=boundary_uncertainty(solution,tracer)
    if not drift["anchor_drift_within_requested_u_tol"]:
        raise ValueError("Fresh weighted C2 anchors exceed the requested root tolerance before screen transport")
    report["maximum_observed_anchor_cdf_uncertainty_estimate"]=drift["maximum_anchor_cdf_uncertainty_estimate"]
    return report


def case_state(snapshot, case, controls, c2_cs):
    state = snapshot.restore()
    lens = next(item for item in state.lenses if item.key == "condenser_lens_2")
    changes = CASES[case]
    lens.cs_mm = c2_cs+changes.get("cs_mm", 0.)
    intrinsic = dict(state.lens_intrinsic_aberrations.get(lens.key, {}))
    for key, value in changes.items():
        if key != "cs_mm":
            intrinsic[key] = intrinsic.get(key, 0.)+value
    state.lens_intrinsic_aberrations[lens.key] = intrinsic
    if {item.key: float(item.percent) for item in state.lenses if item.key in controls} != controls:
        raise AssertionError("The original Image projector controls changed")
    return state


def query_table_arrays(base_points, base_indices, radial_u):
    """Batch unique queries and fixed endpoint anchors, without changing source CDFs.

    Source coordinate order is position radius, position azimuth, angle radius,
    angle azimuth, Young energy, Boersch energy. Anchors are actual zero-current
    probes at u=0 and u=1 for every base, retained in every batch to measure the
    effect of the gun's shared adaptive time step.
    """
    import numpy as np
    base = np.asarray(base_points, dtype=np.float64)
    indices, u = np.asarray(base_indices), np.asarray(radial_u, dtype=np.float64)
    if (base.ndim != 2 or base.shape[1] != 5 or not len(base)
            or np.any(~np.isfinite(base)) or np.any((base < 0) | (base > 1))):
        raise ValueError("Five-dimensional source bases must be finite CDF coordinates")
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or u.shape != indices.shape
            or np.any(indices < 0) or np.any(indices >= len(base))
            or np.any(~np.isfinite(u)) or np.any((u < 0) | (u > 1))):
        raise ValueError("Radial queries require valid base indices and CDF coordinates in [0,1]")
    requested = [(int(index), float(value)) for index, value in zip(indices, u)]
    unique = sorted(set(requested) | {(index, value) for index in range(len(base)) for value in (0., 1.)})
    lookup = {key: index for index, key in enumerate(unique)}
    bi = np.asarray([index for index, _ in unique], dtype=np.int64)
    radial = np.asarray([value for _, value in unique], dtype=np.float64)
    cdf = np.empty((len(unique), 6), dtype=np.float64)
    cdf[:, :2], cdf[:, 2], cdf[:, 3:] = base[bi, :2], radial, base[bi, 2:]
    return dict(coordinates=np.repeat(cdf, 2, axis=0),
        weights=np.zeros(2*len(unique), dtype=np.float64),
        signs=np.tile(np.array([1, -1], dtype=np.int64), len(unique)),
        ray_ids=np.arange(2*len(unique), dtype=np.int64),
        query_base_index=bi, query_radial_u=radial,
        requested_rows=np.asarray([lookup[key] for key in requested], dtype=np.int64),
        anchor_rows=np.asarray([lookup[(index, value)] for index in range(len(base))
                                for value in (0., 1.)], dtype=np.int64))


def explicit_probe_table(arrays, construction):
    from temsim.optics.electron_gun.emitter import EmissionCDFQuadrature
    return EmissionCDFQuadrature.from_arrays(arrays["coordinates"], arrays["weights"],
        signs=arrays["signs"], ray_ids=arrays["ray_ids"], role="boundary-probes",
        construction=construction)


def terminal_arrays(result, target_z_mm):
    """Actual terminal checkpoint and first-stop records, including zero-current rays."""
    import numpy as np
    segment = result.simulation.section_checkpoint.segments[-1]
    cp, branch = segment.checkpoints, segment.branch
    if float(cp.z_mm[-1]) != float(target_z_mm):
        raise AssertionError("Conditional probe target must be an executed exact plane")
    arrays = {key: np.asarray(getattr(cp, key)[-1]) for key in
              ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s")}
    if any(value.dtype != np.float64 for value in arrays.values()):
        raise AssertionError("Conditional probes require float64 solver checkpoints")
    arrays.update(source_id=np.asarray(branch.source_ray_id), weight=np.asarray(branch.ray_weight),
        alive=np.asarray(branch.alive, dtype=bool), blocked_key=np.asarray(branch.blocked_key, dtype=str),
        blocked_z_mm=np.asarray(branch.blocked_z, dtype=np.float64))
    count = len(arrays["source_id"])
    if any(value.shape != (count,) for value in arrays.values()):
        raise AssertionError("Terminal probe arrays lost source-column correspondence")
    return arrays


def c2_margin(arrays, aperture):
    """Signed circular opening margin in mm, checked against actual clipping."""
    import numpy as np
    blocked = arrays["blocked_key"]
    if np.any((blocked != "") & (blocked != aperture.key)):
        raise ValueError("A conditional probe stopped upstream of C2; additional physical boundaries need explicit treatment")
    x_mm, y_mm = arrays["x_m"]*1000., arrays["y_m"]*1000.
    margin = float(aperture.radius_mm)-np.hypot(x_mm-float(aperture.offset_x_mm),
                                             y_mm-float(aperture.offset_y_mm))
    if np.any(~np.isfinite(margin)):
        raise ValueError("The C2 opening margin requires actual finite arrival coordinates")
    if not np.array_equal(margin >= 0., arrays["alive"]):
        raise AssertionError("Terminal float64 opening margin disagrees with production aperture clipping")
    stopped = blocked == aperture.key
    if np.any(arrays["blocked_z_mm"][stopped] != float(aperture.z_mm)):
        raise AssertionError("C2 stop records do not identify its physical plane")
    return margin


def configure_inputs(reference, backend, step_mm):
    """Restore historical parameters only; never install historical trajectories."""
    from temsim.input_design import restore_input_design
    from temsim.instrument_snapshot import InstrumentSnapshot
    from temsim.component_keys import FLUORESCENT_SCREEN
    from temsim.physics.aperture_clipping import posed_aperture_registration
    reference = Path(reference)
    original = InstrumentSnapshot.from_dict(json.loads((reference/"baseline.snapshot.json").read_text()))
    report = json.loads((reference/"comparison.json").read_text())
    state = restore_input_design(original)
    if state.sample.inserted or state.sample.specimen_mode != "vacuum" or state.vacuum_map.enabled:
        raise ValueError("Conditional diagnostic is limited to the recorded vacuum optical path")
    if (not state.fluorescent_screen.inserted or any(receiver.inserted and receiver.key != FLUORESCENT_SCREEN
                                                    for receiver in state.recording_planes)):
        raise ValueError("The original physical main screen must be the only inserted receiver")
    emitter = state.electron_gun.emitter
    if emitter.surface_model is not None or emitter.coherence is not None or emitter.curvature_nm_inv != 0.:
        raise ValueError("This conditional source diagnostic requires the recorded flat classical source")
    controls = report["projector_controls"]["imaging"]["lens_percent"]
    for lens in state.lenses:
        if lens.key in controls:
            lens.percent = controls[lens.key]
    c2_cs = float(report["baseline_c2_cs_mm"])
    next(lens for lens in state.lenses if lens.key == "condenser_lens_2").cs_mm = c2_cs
    state.acceleration_enabled, state.acceleration_backend = True, backend
    state.projector_mode, state.step_mm = "image", step_mm
    aperture = state.condenser_aperture_2
    if not aperture.enabled or aperture.radius_mm <= 0 or posed_aperture_registration(state, aperture) is not None:
        raise ValueError("The conditional margin currently requires the active, unposed circular C2 aperture")
    return state, controls, c2_cs


class PhysicalBoundaryTracer:
    """One fresh, explicit zero-current source per globally batched root query."""

    def __init__(self, state, base_points, output, manifest, controls, c2_cs, base_plan=None):
        from temsim.instrument_snapshot import capture_instrument_snapshot
        self.snapshot = capture_instrument_snapshot(state)
        self.base_points, self.output, self.manifest = base_points, Path(output), manifest
        self.controls, self.c2_cs, self.calls = controls, c2_cs, 0
        self.base_plan = base_plan
        self.anchor_reference = None
        self.anchor_drift = None

    def guard(self):
        from temsim.calculation_manifest import solver_source_identity
        if solver_source_identity() != self.manifest["implementation"]:
            raise RuntimeError("Production implementation changed during conditional source execution")
        protocol=self.manifest["protocol"]
        if digest_file(Path(__file__))!=protocol["script_sha256"]:
            raise RuntimeError("Conditional collector changed during execution")
        for name,sha in protocol["helper_sha256"].items():
            if digest_file(ROOT/"scripts"/name)!=sha:
                raise RuntimeError("Conditional numerical helper changed during execution")

    def __call__(self, base_indices, radial_u):
        import numpy as np
        from temsim.instrument_snapshot import capture_instrument_snapshot
        from temsim.particle_section_io import save_section_result
        from temsim.physics.backend_execution import backend_receipts
        from temsim.simulation_pipeline import calculate_particle_section
        self.guard()
        arrays = query_table_arrays(self.base_points, base_indices, radial_u)
        table = explicit_probe_table(arrays, "Zero-current conditional CDF boundary probes; "
                                     + canonical_digest(dict(base=array_digest({"points": self.base_points}),
                                                             query=array_digest({key:value for key,value in arrays.items()
                                                                                 if key != "requested_rows"}))))
        state = self.snapshot.restore()
        emitter = state.electron_gun.emitter
        emitter.sampling = None
        emitter.cdf_quadrature = table
        emitter.ray_count = table.total
        snapshot = capture_instrument_snapshot(state)
        identity = canonical_digest(dict(protocol=self.manifest["protocol_digest"],
            snapshot=snapshot.digest, table=canonical_digest(asdict(table))))
        self.calls += 1
        for name, stored in self.manifest["batches"].items():
            if stored.get("request_identity") == identity and stored["status"] == "COMPLETE":
                directory = self.output/name
                for relative, sha in stored["files"].items():
                    if digest_file(directory/relative) != sha:
                        raise ValueError(f"Cached physical callback file changed: {name}/{relative}")
                with np.load(directory/"margins.npz", allow_pickle=False) as data:
                    values, anchors = data["margin_mm"], data["anchor_margin_mm"]
                self.observe_anchors(anchors)
                print(f"{name}: verified completed physical callback reused", flush=True)
                return values[arrays["requested_rows"]]
        directory = self.output / f"batch-{len(self.manifest['batches'])+1:03d}"
        directory.mkdir()
        write_json(directory/"input.snapshot.json", snapshot.to_dict())
        write_npz(directory/"queries.npz", arrays)
        record = dict(status="RUNNING", snapshot_digest=snapshot.digest,
            request_identity=identity,
            snapshot_sha256=digest_file(directory/"input.snapshot.json"),
            table_digest=canonical_digest(asdict(table)), rays=table.total,
            requested_queries=len(base_indices), unique_queries=len(arrays["query_base_index"]),
            role="boundary-probes", total_probability=float(arrays["weights"].sum()), cases={})
        self.manifest["batches"][directory.name] = record
        write_json(self.output/"manifest.json", self.manifest)
        receipts = []
        started = perf_counter()
        with backend_receipts(receipts):
            seed = calculate_particle_section(state, state.electron_gun.exit_plane_z_mm+.01,
                                              acquire_stem=False)
        self.guard()
        source = source_arrays(seed)
        if not np.array_equal(source["gun_exit_weight"], arrays["weights"]):
            raise AssertionError("Zero-current boundary probes acquired physical statistical weight")
        if not np.all(source["gun_exit_alive"]):
            raise ValueError("Conditional source probes include a physical gun stop; its boundary must be treated explicitly")
        source_digest = array_digest(source)
        write_npz(directory/"source.npz", source)
        save_section_result(seed, directory/"source.temsection", overwrite=False)
        record.update(gun_runtime_s=perf_counter()-started, gun_backend_receipts=receipts,
                      source_digest=source_digest, source_sha256=digest_file(directory/"source.npz"),
                      nominal_configured_current_a=float(seed.simulation.gun_trace.emitted_current_a),
                      weighted_probe_current_a=float(seed.simulation.gun_trace.emitted_current_a)*float(source["gun_exit_weight"].sum()))
        write_json(self.output/"manifest.json", self.manifest)
        print(f"{directory.name}: {table.total} zero-current probes; fresh gun {record['gun_runtime_s']:.2f}s", flush=True)
        margins, previous = [], seed
        for case, changes in CASES.items():
            run_state = case_state(snapshot, case, self.controls, self.c2_cs)
            target = float(run_state.condenser_aperture_2.z_mm)
            self.guard()
            receipts, started = [], perf_counter()
            with backend_receipts(receipts):
                result = calculate_particle_section(run_state, target, existing_result=previous, acquire_stem=False)
            self.guard()
            if (array_digest(source_arrays(result)) != source_digest
                    or not result.simulation.metrics["section_gun_reused"]):
                raise AssertionError("Cases did not share the same actual, identity-checked physical gun")
            end = terminal_arrays(result, target)
            if (not np.array_equal(end["source_id"], arrays["ray_ids"])
                    or not np.array_equal(end["weight"], arrays["weights"])):
                raise AssertionError("C2 probes changed their explicit source IDs or zero weights")
            margin = c2_margin(end, run_state.condenser_aperture_2)
            margins.append(margin.reshape(-1, 2))
            write_npz(directory/f"{case}.npz", dict(**end, margin_mm=margin))
            run_snapshot = capture_instrument_snapshot(run_state)
            write_json(directory/f"{case}.snapshot.json", run_snapshot.to_dict())
            record["cases"][case] = dict(runtime_s=perf_counter()-started, backend_receipts=receipts,
                target_z_mm=target, source_digest=source_digest, snapshot_digest=run_snapshot.digest,
                particles_sha256=digest_file(directory/f"{case}.npz"),
                passed_rays=int(end["alive"].sum()), current_probability=0.)
            previous = result
            write_json(self.output/"manifest.json", self.manifest)
        values = np.concatenate(margins, axis=1)
        anchors = values[arrays["anchor_rows"]]
        self.observe_anchors(anchors)
        record.update(status="COMPLETE", implementation_after=self.manifest["implementation"],
            anchor_margin_max_change_mm=float(np.max(np.abs(anchors-self.anchor_reference))))
        write_npz(directory/"margins.npz", dict(margin_mm=values, anchor_margin_mm=anchors))
        record["files"] = {path.name:digest_file(path) for path in directory.iterdir() if path.is_file()}
        write_json(self.output/"manifest.json", self.manifest)
        print(f"{directory.name}: four physical C2 cases complete; zero predicted exposure", flush=True)
        return values[arrays["requested_rows"]]

    def observe_anchors(self, anchors):
        import numpy as np
        if self.anchor_reference is None:
            self.anchor_reference = anchors.copy()
            self.anchor_drift = np.zeros_like(anchors)
        self.anchor_drift = np.maximum(self.anchor_drift, np.abs(anchors-self.anchor_reference))
        if getattr(self,"base_plan",None) is not None:
            check=sampled_energy_containment(anchors,self.base_plan["mapping"],CHANNELS)
            self.manifest["energy_containment"]=check
            write_json(self.output/"manifest.json",self.manifest)
            if check["failure_count"]:
                raise ValueError("Actual endpoint anchors show an energy support transition outside the declared band")


def quadrature_arrays(base, table):
    """Full-support CDF Gauss measure, with explicit half-probability signs."""
    import numpy as np
    cdf = np.empty((len(table.weight),6), dtype=np.float64)
    cdf[:,:2], cdf[:,2], cdf[:,3:] = base[table.base_index,:2], table.angular_cdf, base[table.base_index,2:]
    anchors=query_table_arrays(base,np.array([],dtype=np.int64),np.array([],dtype=np.float64))
    return dict(coordinates=np.concatenate((np.repeat(cdf,2,axis=0),anchors["coordinates"])),
        weights=np.r_[np.repeat(table.weight/2,2),anchors["weights"]],
        signs=np.r_[np.tile(np.array([1,-1],dtype=np.int64),len(cdf)),anchors["signs"]],
        ray_ids=np.arange(2*len(cdf)+len(anchors["weights"]),dtype=np.int64),
        base_index=np.r_[np.repeat(table.base_index,2),np.repeat(anchors["query_base_index"],2)],
        angular_cdf=np.r_[np.repeat(table.angular_cdf,2),np.repeat(anchors["query_radial_u"],2)],
        cell_index=np.r_[np.repeat(table.cell_index,2),np.full(len(anchors["weights"]),-1,dtype=np.int64)],
        expected_pass=table.expected_pass)


def boundary_uncertainty(solution, tracer):
    """Observed batch drift / local slope; an estimate, not a certified bound."""
    import numpy as np
    rows = []
    for root in solution.roots:
        b,c,x = root["base_index"],root["channel_index"],root["u"]
        samples = sorted((u,float(v[c])) for (index,u),v in solution.evaluator.cache.items() if index==b)
        left = [(u,v) for u,v in samples if u<x]
        right = [(u,v) for u,v in samples if u>x]
        if not left or not right:
            raise ValueError("A boundary has no actual two-sided slope estimate")
        a,fa=left[-1]; z,fz=right[0]
        slope = abs((fz-fa)/(z-a))
        if not np.isfinite(slope) or slope<=0:
            raise ValueError("A boundary has no finite nonzero actual slope")
        drift = float(np.max(tracer.anchor_drift.reshape(-1,2,len(CHANNELS))[b,:,c]))
        rows.append(dict(**root, local_slope_mm_per_cdf=float(slope),
            observed_anchor_drift_mm=drift, anchor_cdf_uncertainty_estimate=drift/slope,
            bracket_plus_anchor_estimate=root["bracket_width"]+drift/slope))
    pairs=[]
    for b in range(len(solution.partitions)):
        for sign in (1,-1):
            baseline=[r for r in rows if r["base_index"]==b and r["channel_key"]==f"baseline:{sign:+d}"]
            for case in list(CASES)[1:]:
                varied=[r for r in rows if r["base_index"]==b and r["channel_key"]==f"{case}:{sign:+d}"]
                if len(baseline)==len(varied)==1:
                    a,z=baseline[0],varied[0]
                    shift=z["u"]-a["u"]
                    uncertainty=a["bracket_plus_anchor_estimate"]+z["bracket_plus_anchor_estimate"]
                    pairs.append(dict(base_index=b,case=case,sign=sign,root_shift=shift,
                        bracket_plus_anchor_uncertainty_estimate=uncertainty,
                        response_resolved=abs(shift)>uncertainty))
    maximum=max((r["anchor_cdf_uncertainty_estimate"] for r in rows),default=0.)
    return dict(roots=rows,paired_root_shifts=pairs,maximum_anchor_cdf_uncertainty_estimate=maximum,
        maximum_anchor_margin_drift_mm=float(np.max(tracer.anchor_drift)),
        anchor_drift_within_requested_u_tol=maximum<=solution.diagnostics["angular_cdf_tolerance"],
        scope="Observed fixed endpoint drift divided by actual local slope; not a bound on unobserved interior drift or absolute transport error.")


def collect_screen_group(tracer, solution, table, order, partition_digest):
    """Emit the full positive-weight table freshly, then share that gun in four cases."""
    import numpy as np
    from temsim.optics.electron_gun.emitter import EmissionCDFQuadrature
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.particle_section_io import load_section_result, save_section_result
    from temsim.physics.backend_execution import backend_receipts
    from temsim.simulation_pipeline import calculate_particle_section
    manifest, output = tracer.manifest, tracer.output
    group=group_label(manifest["base_count"],manifest.get("azimuth_orbit_size",2),order, "energy_stratification" in manifest)
    directory=output/group
    directory.mkdir(exist_ok=True)
    if 2*len(table.weight)+4*len(tracer.base_points)>1048576:
        raise ValueError("Final actual source table exceeds the explicit source limit")
    arrays=quadrature_arrays(tracer.base_points,table)
    half=manifest.get("azimuth_orbit_size",2)//2
    arrays["independent_base_index"]=arrays["base_index"]//half
    arrays["half_orbit_index"]=arrays["base_index"]%half
    if getattr(tracer,"base_plan",None) is not None:
        for key,value in tracer.base_plan["mapping"].items():
            arrays[key]=value[arrays["base_index"]]
    explicit=EmissionCDFQuadrature.from_arrays(arrays["coordinates"],arrays["weights"],
        signs=arrays["signs"],ray_ids=arrays["ray_ids"],role="quadrature",
        construction=f"Full radial CDF union-cell Gauss {order}; partition {partition_digest}")
    table_digest=canonical_digest(asdict(explicit))
    state=tracer.snapshot.restore()
    emitter=state.electron_gun.emitter
    emitter.sampling=None; emitter.cdf_quadrature=explicit; emitter.ray_count=explicit.total
    snapshot=capture_instrument_snapshot(state)
    stored=manifest["sources"].get(group)
    tracer.guard()
    if stored:
        if stored["source_table_digest"]!=table_digest or stored["snapshot_digest"]!=snapshot.digest:
            raise ValueError("Retained quadrature source table changed")
        for pathkey,shakey in (("source_path","source_sha256"),("archive_path","archive_sha256"),
                              ("table_path","table_sha256"),("table_record_path","table_record_sha256"),
                              ("snapshot_path","snapshot_sha256")):
            if digest_file(output/stored[pathkey])!=stored[shakey]:
                raise ValueError("Retained quadrature source evidence changed")
        seed=load_section_result(output/stored["archive_path"])
        source=source_arrays(seed)
        if array_digest(source)!=stored["source_digest"]:
            raise ValueError("Retained actual quadrature gun changed")
    else:
        write_npz(directory/"table.npz",arrays)
        write_json(directory/"table.json",asdict(explicit))
        write_json(directory/"input.snapshot.json",snapshot.to_dict())
        started,receipts=perf_counter(),[]
        with backend_receipts(receipts):
            seed=calculate_particle_section(state,state.electron_gun.exit_plane_z_mm+.01,acquire_stem=False)
        tracer.guard()
        source=source_arrays(seed)
        if (not np.array_equal(source["gun_exit_ray_id"],arrays["ray_ids"])
                or not np.array_equal(source["gun_exit_weight"],arrays["weights"])):
            raise AssertionError("Actual quadrature gun altered source probability or ancestry")
        write_npz(directory/"source.npz",source)
        save_section_result(seed,directory/"source.temsection",overwrite=(directory/"source.temsection").exists())
        stored=dict(base_count=manifest["base_count"],expanded_base_count=len(tracer.base_points),
            azimuth_orbit_size=manifest.get("azimuth_orbit_size",2),gauss_order=order,rays=explicit.total,
            source_table_digest=table_digest,source_digest=array_digest(source),partition_digest=partition_digest,
            snapshot_digest=snapshot.digest,snapshot_path=f"{group}/input.snapshot.json",
            snapshot_sha256=digest_file(directory/"input.snapshot.json"),
            source_path=f"{group}/source.npz",source_sha256=digest_file(directory/"source.npz"),
            archive_path=f"{group}/source.temsection",archive_sha256=digest_file(directory/"source.temsection"),
            table_path=f"{group}/table.npz",table_sha256=digest_file(directory/"table.npz"),
            table_record_path=f"{group}/table.json",table_record_sha256=digest_file(directory/"table.json"),
            runtime_s=perf_counter()-started,backend_receipts=receipts,
            emitted_current_a=float(seed.simulation.gun_trace.emitted_current_a),
            total_probability=float(source["gun_exit_weight"].sum()),gun_survivors=int(source["gun_exit_alive"].sum()))
        manifest["sources"][group]=stored
        write_json(output/"manifest.json",manifest)
        print(f"{group}: fresh full-support physical gun {explicit.total} rays, {stored['runtime_s']:.2f}s",flush=True)
    if not np.all(source["gun_exit_alive"]):
        raise ValueError("Final quadrature contains gun stops absent from the solved C2 partition")
    previous=seed
    final_anchors=[]
    for case_index,case in enumerate(CASES):
        run_id=f"{group}_{case}"
        run_state=case_state(snapshot,case,tracer.controls,tracer.c2_cs)
        target=float(run_state.fluorescent_screen.z_mm)
        run_snapshot=capture_instrument_snapshot(run_state)
        request=canonical_digest(dict(protocol=manifest["protocol_digest"],state=run_snapshot.digest,
            target_z_mm=target,run_id=run_id,source_table=table_digest,partition=partition_digest))
        retained=manifest["runs"].get(run_id)
        if retained and retained["status"]=="COMPLETE":
            if (retained["request_identity"]!=request
                    or digest_file(output/retained["particles_path"])!=retained["particles_sha256"]
                    or digest_file(output/retained["snapshot_path"])!=retained["snapshot_sha256"]
                    or digest_file(output/retained["c2_path"])!=retained["c2_sha256"]):
                raise ValueError("Completed screen quadrature identity changed")
            with np.load(output/retained["c2_path"],allow_pickle=False) as data:
                c2values=data["margin_mm"]
            expected=table.expected_pass[:,2*case_index:2*case_index+2].reshape(-1)
            if not np.array_equal(c2values[:len(expected)]>=0.,expected):
                raise ValueError("Retained actual C2 membership changed")
            final_anchors.append(c2values[len(expected):].reshape(-1,2))
            print(f"{run_id}: verified completed screen retained",flush=True)
            continue
        write_json(directory/f"{case}.snapshot.json",run_snapshot.to_dict())
        row=dict(status="RUNNING",base_count=manifest["base_count"],expanded_base_count=len(tracer.base_points),
            azimuth_orbit_size=manifest.get("azimuth_orbit_size",2),gauss_order=order,case=case,
            source_table_digest=table_digest,source_digest=stored["source_digest"],partition_digest=partition_digest,
            request_identity=request,implementation_before=manifest["implementation"],projector_controls=tracer.controls,
            receiver_z_mm=target,particles_path=f"{group}/{case}.npz",
            snapshot_path=f"{group}/{case}.snapshot.json",snapshot_sha256=digest_file(directory/f"{case}.snapshot.json"))
        manifest["runs"][run_id]=row
        write_json(output/"manifest.json",manifest)
        started,receipts=perf_counter(),[]
        tracer.guard()
        with backend_receipts(receipts):
            c2result=calculate_particle_section(run_state,float(run_state.condenser_aperture_2.z_mm),
                                               existing_result=previous,acquire_stem=False)
            c2end=terminal_arrays(c2result,float(run_state.condenser_aperture_2.z_mm))
            c2values=c2_margin(c2end,run_state.condenser_aperture_2)
            expected=table.expected_pass[:,2*case_index:2*case_index+2].reshape(-1)
            if not np.array_equal(c2values[:len(expected)]>=0.,expected):
                write_npz(directory/f"{case}.membership-failed.npz",dict(**c2end,margin_mm=c2values))
                raise ValueError("Fresh full-support gun/column disagrees with conditional C2 membership")
            case_anchors=c2values[len(expected):].reshape(-1,2)
            final_anchors.append(case_anchors)
            write_npz(directory/f"{case}.c2.npz",dict(**c2end,margin_mm=c2values))
            if getattr(tracer,"base_plan",None) is not None:
                check=validate_weighted_case_anchors(tracer,solution,case_anchors,case_index,CHANNELS)
                row["weighted_source_containment"]=check
                write_json(output/"manifest.json",manifest)

            result=calculate_particle_section(run_state,target,existing_result=c2result,acquire_stem=False)
        tracer.guard()
        if (array_digest(source_arrays(result))!=stored["source_digest"]
                or not result.simulation.metrics["section_gun_reused"]):
            raise AssertionError("Screen cases changed the actual common gun")
        end=screen_arrays(result,target,source,stored["emitted_current_a"])
        actual_c2_pass=end["blocked_key"]!=run_state.condenser_aperture_2.key
        if not np.array_equal(actual_c2_pass[:len(expected)],expected):
            write_npz(directory/f"{case}.membership-failed.npz",end)
            raise ValueError("Fresh full-support gun/column disagrees with conditional C2 membership")
        write_npz(directory/f"{case}.npz",end)
        labels,counts=np.unique(end["blocked_key"],return_counts=True)
        row.update(status="COMPLETE",implementation_after=manifest["implementation"],runtime_s=perf_counter()-started,
            backend_receipts=receipts,particles_sha256=digest_file(directory/f"{case}.npz"),
            screen_hits=int(end["survivor"].sum()),screen_probability=float(end["weight"][end["survivor"]].sum()),
            stop_counts=dict(zip(labels.tolist(),counts.tolist())),actual_c2_membership_matches=True,
            c2_path=f"{group}/{case}.c2.npz",c2_sha256=digest_file(directory/f"{case}.c2.npz"))
        write_json(output/"manifest.json",manifest)
        previous=result
        print(f"{run_id}: actual screen complete {row['runtime_s']:.2f}s, hits {row['screen_hits']}",flush=True)
    if len(final_anchors)==len(CASES):
        anchors=np.concatenate(final_anchors,axis=1)
        tracer.observe_anchors(anchors)
        stored["final_anchor_max_drift_mm"]=float(np.max(np.abs(anchors-tracer.anchor_reference)))
        write_npz(directory/"anchor-margins.npz",dict(anchor_margin_mm=anchors))
        stored.update(anchor_path=f"{group}/anchor-margins.npz",anchor_sha256=digest_file(directory/"anchor-margins.npz"))
        write_json(output/"manifest.json",manifest)


def main():
    args = arguments()
    for key in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import numpy as np
    from scipy.stats import qmc
    from temsim.calculation_manifest import solver_source_identity
    from temsim.cpu_resources import numerical_job
    from temsim.detector.point_spread import DetectorPointSpread
    from temsim.instrument_snapshot import capture_instrument_snapshot
    output = args.output.resolve()
    state, controls, c2_cs = configure_inputs(args.reference, args.backend, args.step)
    gun_steps=apply_gun_step_scale(state.electron_gun,args.gun_step_scale)
    base = qmc.Sobol(d=5, scramble=True, seed=args.seed, bits=30).random_base2(args.base_count.bit_length()-1)
    from scripts.source_azimuth_orbit import expand_common_azimuth
    orbit=expand_common_azimuth(base,args.azimuth_orbit_size)
    expanded_base=orbit.base_points
    base_plan=None
    if args.energy_band is not None:
        base_plan=compose_energy_orbit(state.electron_gun.emitter,base,args.azimuth_orbit_size,args.energy_band)
        expanded_base=base_plan["points"]
    mapping=(base_plan["mapping"] if base_plan is not None else
             dict(independent_base_index=orbit.independent_base_index,half_orbit_index=orbit.half_orbit_index))
    base_snapshot=capture_instrument_snapshot(state)
    physical_protocol=dict(input_snapshot=base_snapshot.digest,step_mm=args.step,backend=args.backend,
        source_schema="tip-explicit-cdf-v1",azimuth_orbit_schema=orbit.metadata["schema"],source_base_seed=args.seed,source_base_dimensions=5,
        projector_controls=controls,cases=CASES,c2_cs_mm=c2_cs,
        nominal_current_a=float(state.electron_gun.emitted_current_a),gun_integration=gun_steps)
    helper_names=["source_radial_quadrature.py","source_azimuth_orbit.py","validate_image_intensity_convergence.py"]
    if base_plan is not None:
        physical_protocol["energy_stratification"]=dict(schema=base_plan["energy_metadata"]["schema"],
            band_ev=list(args.energy_band),cuts_metadata=base_plan["cuts_metadata"])
        # Execution optimization is protocol-bound but not part of the source law.
        helper_names.append("source_energy_strata.py")
    protocol=dict(physical_protocol=physical_protocol,implementation=solver_source_identity(),
        script_sha256=digest_file(Path(__file__)),base_count=args.base_count,stage=args.stage,
        helper_sha256={name:digest_file(ROOT/"scripts"/name) for name in
                      helper_names},
        scan_points=args.scan_points,scan_intervals=args.scan_intervals,scan_refinements=args.scan_refinements,
        u_tol=args.u_tol,maxiter=args.maxiter,azimuth_orbit_size=args.azimuth_orbit_size,
        azimuth_orbit_schema=orbit.metadata["schema"],orders=sorted(set(args.orders)),threads=args.threads)
    if base_plan is not None:
        protocol["gauss_validation_execution"]="fresh-weighted-gun-C2-membership-and-anchors-before-screen-v1"
    manifest = dict(schema=SCHEMA, status="PREPARED", implementation=solver_source_identity(),
        script_sha256=digest_file(Path(__file__)), stage=args.stage, base_count=args.base_count,
        base_seed=args.seed,expanded_base_count=len(expanded_base),azimuth_orbit_size=args.azimuth_orbit_size,
        azimuth_orbit=orbit.metadata,step_mm=args.step, scan_points=args.scan_points, channels=CHANNELS,
        backend=args.backend, threads=args.threads, projector_controls=controls, c2_cs_mm=c2_cs,
        aperture_z_mm=float(state.condenser_aperture_2.z_mm),
        receiver_z_mm=float(state.fluorescent_screen.z_mm),
        point_spread=asdict(DetectorPointSpread.from_component(state.fluorescent_screen)),
        base_digest=array_digest({"points": base}), batches={},runs={},sources={},
        protocol=protocol,protocol_digest=canonical_digest(protocol),physical_protocol=physical_protocol,
        physical_protocol_digest=canonical_digest(physical_protocol),base_points_path="base-points.npz",
        expanded_base_points_path="expanded-base-points.npz",
        expanded_base_digest=array_digest({"points":expanded_base}),
        azimuth_orbit_map_path="azimuth-orbit-map.npz",
        scope="Actual zero-current tip-to-C2 boundary probes and fresh full-support quadrature sources through the physical Image screen. Finite scanning does not certify absence of arbitrarily narrow intervals; a single base count or Gauss order does not establish convergence.")
    if base_plan is not None:
        manifest.update(energy_stratification=base_plan["energy_metadata"],azimuth_orbit=base_plan["orbit_metadata"],
            source_base_measure_path="source-base-measure.npz",source_base_measure_digest=array_digest(mapping),
            energy_cuts_path="energy-cuts.npz",energy_cuts_digest=array_digest(base_plan["cuts"]),
            energy_cuts_metadata=base_plan["cuts_metadata"])
    if output.exists():
        if not args.resume:
            raise ValueError("Existing evidence requires explicit --resume with identical protocol")
        saved=json.loads((output/"manifest.json").read_text())
        validate_resume_metadata(manifest,saved)
        if saved["protocol_digest"]!=manifest["protocol_digest"] or canonical_digest(saved["protocol"])!=manifest["protocol_digest"]:
            raise ValueError("Resume requires exactly the executed source, optical, solver and collector identities")
        if digest_file(output/"base-points.npz")!=saved["base_points_sha256"]:
            raise ValueError("Retained five-dimensional source bases changed")
        for name,sha in protocol["helper_sha256"].items():
            if digest_file(output/"helpers"/name)!=sha:
                raise ValueError("Retained conditional helper changed")
        with np.load(output/"base-points.npz",allow_pickle=False) as data:
            if not np.array_equal(data["points"],base):
                raise ValueError("Retained original source coordinates differ from the actual requested sequence")
        if digest_file(output/"expanded-base-points.npz")!=saved["expanded_base_points_sha256"]:
            raise ValueError("Retained expanded azimuth source bases changed")
        with np.load(output/"expanded-base-points.npz",allow_pickle=False) as data:
            if not np.array_equal(data["points"],expanded_base):
                raise ValueError("Retained expanded source coordinates differ from the requested orbit")
        if digest_file(output/"azimuth-orbit-map.npz")!=saved["azimuth_orbit_map_sha256"]:
            raise ValueError("Retained source azimuth mapping changed")
        with np.load(output/"azimuth-orbit-map.npz",allow_pickle=False) as data:
            if any(not np.array_equal(data[key],value) for key,value in mapping.items()):
                raise ValueError("Retained source azimuth mapping differs from the requested orbit")
        if base_plan is not None:
            for stem,values in (("source_base_measure",mapping),("energy_cuts",base_plan["cuts"])):
                path=output/saved[stem+"_path"]
                if digest_file(path)!=saved[stem+"_sha256"]:
                    raise ValueError("Retained energy measure archive changed")
                with np.load(path,allow_pickle=False) as data:
                    if set(data.files)!=set(values) or any(not np.array_equal(data[k],v) for k,v in values.items()):
                        raise ValueError("Retained energy measure differs from the requested exact CDF partition")
        manifest=saved
    else:
        output.mkdir(parents=True)
        write_npz(output/"base-points.npz", {"points": base})
        manifest["base_points_sha256"]=digest_file(output/"base-points.npz")
        write_npz(output/"expanded-base-points.npz",{"points":expanded_base})
        manifest["expanded_base_points_sha256"]=digest_file(output/"expanded-base-points.npz")
        write_npz(output/"azimuth-orbit-map.npz",mapping)
        manifest["azimuth_orbit_map_sha256"]=digest_file(output/"azimuth-orbit-map.npz")
        manifest["azimuth_orbit_map_digest"]=array_digest(mapping)
        if base_plan is not None:
            for stem,values in (("source_base_measure",mapping),("energy_cuts",base_plan["cuts"])):
                path=output/manifest[stem+"_path"]
                write_npz(path,values)
                manifest[stem+"_sha256"]=digest_file(path)
        write_json(output/"base-input.snapshot.json",base_snapshot.to_dict())
        (output/"collector.executed.py").write_bytes(Path(__file__).read_bytes())
        (output/"helpers").mkdir()
        for name in protocol["helper_sha256"]:
            (output/"helpers"/name).write_bytes((ROOT/"scripts"/name).read_bytes())
        write_json(output/"source-inventory.json",{str(path.relative_to(ROOT)):digest_file(path)
            for path in sorted((ROOT/"src").rglob("*.py"))})
    write_json(output/"manifest.json", manifest)
    if args.stage == "prepare":
        print(json.dumps({"status": "PREPARED", "base_count":args.base_count,
                          "initial_full_scan_probe_rays":2*len(expanded_base)*(args.scan_intervals*2**args.scan_refinements+1),
                          "output":str(output)}), flush=True)
        return
    started = perf_counter()
    tracer = PhysicalBoundaryTracer(state, expanded_base, output, manifest, controls, c2_cs,base_plan=base_plan)
    manifest["status"]="RUNNING"
    write_json(output/"manifest.json",manifest)
    try:
        with numerical_job(args.threads):
            if args.stage=="coarse":
                index = np.repeat(np.arange(len(expanded_base)), args.scan_points)
                u = np.tile(np.linspace(0., 1., args.scan_points), len(expanded_base))
                margin = tracer(index, u).reshape(len(expanded_base), args.scan_points, len(CHANNELS))
                crossing_count = np.count_nonzero((margin[:, 1:] >= 0.) != (margin[:, :-1] >= 0.), axis=1)
                write_npz(output/"coarse-scan.npz", dict(base_index=index, radial_u=u,
                    margin_mm=margin, sampled_crossing_count=crossing_count))
                manifest.update(status="COARSE_SCAN_COMPLETE",sampled_max_crossings=int(crossing_count.max()),
                    sampled_nonincreasing_margin=bool(np.all(np.diff(margin,axis=1)<=0.)))
            else:
                from scripts.source_radial_quadrature import solve_radial_partitions,gauss_table,validate_node_membership
                solution=solve_radial_partitions(tracer,len(expanded_base),CHANNELS,scan_intervals=args.scan_intervals,
                    scan_refinements=args.scan_refinements,u_tol=args.u_tol,maxiter=args.maxiter)
                partition=dict(roots=solution.roots,partitions=[dict(base_index=part.base_index,
                    edges=part.edges.tolist(),midpoint_margins=part.midpoint_margins.tolist(),
                    diagnostics=part.diagnostics) for part in solution.partitions],diagnostics=solution.diagnostics)
                if base_plan is not None:
                    partition["weighted_root_budget"]=weighted_root_budget(solution.roots,mapping["base_probability"],CHANNELS)
                    partition["uniform_helper_budget_scope"]="Helper diagnostic assumes uniform rows; use weighted_root_budget for the physical probability measure"
                partition_digest=canonical_digest(partition)
                write_json(output/"partition.json",partition)
                manifest.update(partition_path="partition.json",partition_sha256=digest_file(output/"partition.json"),
                    partition_digest=partition_digest,root_diagnostics=solution.diagnostics,
                    batch_drift=boundary_uncertainty(solution,tracer))
                write_json(output/"manifest.json",manifest)
                if not manifest["batch_drift"]["anchor_drift_within_requested_u_tol"]:
                    raise ValueError("Observed physical gun batch drift exceeds the requested CDF root tolerance")
                for order in sorted(set(args.orders)):
                    table=gauss_table(solution,order=order)
                    if base_plan is not None:
                        table=reweight_radial_table(table,mapping["base_probability"])
                    if base_plan is None:
                        validate_node_membership(solution,table)
                    # Energy protocol validates every node against the fresh weighted
                    # physical gun at C2 below; no duplicate zero-dose validation gun.
                    # Root probes and their endpoint checks remain unchanged.
                    manifest["batch_drift"]=boundary_uncertainty(solution,tracer)
                    if not manifest["batch_drift"]["anchor_drift_within_requested_u_tol"]:
                        raise ValueError("Gauss validation batch drift exceeds the requested CDF root tolerance")
                    collect_screen_group(tracer,solution,table,order,partition_digest)
                    manifest["batch_drift"]=boundary_uncertainty(solution,tracer)
                    if not manifest["batch_drift"]["anchor_drift_within_requested_u_tol"]:
                        raise ValueError("Final weighted source batch drift exceeds the requested CDF root tolerance")
                manifest.update(status="COMPLETE",qualification="SINGLE_BASE_COUNT_CONDITIONAL_QUADRATURE_NOT_TOTAL_CONVERGENCE")
        manifest["last_invocation_elapsed_s"]=perf_counter()-started
        if not args.resume or "elapsed_s" not in manifest:
            manifest["elapsed_s"]=manifest["last_invocation_elapsed_s"]
        write_json(output/"manifest.json", manifest)
        print(json.dumps({"status":manifest["status"], "elapsed_s":manifest["last_invocation_elapsed_s"],
                          "complete_screen_runs":len(manifest["runs"])}), flush=True)
    except Exception as exc:
        manifest.update(status="ERROR", error=f"{type(exc).__name__}: {exc}", elapsed_s=perf_counter()-started)
        write_json(output/"manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
