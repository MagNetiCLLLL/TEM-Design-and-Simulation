"""CUDA ideal-single-mode development benchmark; NOT a physical FEG solution.

The deliberately prescribed Gaussian entrance has 0.4 electron probability per
reference electron. 1024 or more original positive-current particles sample
the SAME ideal Gaussian Wigner source and are traced as a separate classical
reference. Neither frame packets nor exit resampling count as source particles.
The real Si CIF, finite thickness and downstream receiver exercise the hybrid
operators. The field-free short column is not a qualified microscope setting.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.constants import hbar

from temsim.hybrid_cli import save_result
from temsim.physics.hybrid_pipeline import HybridGrid, HybridRequest, HybridResult, _column, _material, _downstream
from temsim.physics.hybrid_semiclassical import SemiclassicalNumerics, momentum
from temsim.physics.hybrid_exit import sample_exit


def restore_ideal_stage(path, source, settings, schema, plane_z_mm):
    """Restore a committed stage of this explicitly ideal development source."""
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dependency = manifest["dependency"]
    downstream = {"output", "resume_incident_manifest", "resume_exit_manifest", "exit_particles", "exit_window_nm"}
    if (dependency.get("scope") != "ideal-hybrid-development-intermediate"
            or dependency.get("source") != source.digest
            or any(settings.get(key) != value for key, value in dependency["settings"].items() if key not in downstream)):
        raise ValueError("Saved ideal stage does not match these declared upstream inputs")
    store = ExecutedWaveStore(path.resolve().parent.parent, dependency, 2*1024**3)
    stage = store.get(manifest["key"])
    if stage is None or stage.plane_z_mm != plane_z_mm or stage.record["schema"] != schema:
        raise ValueError("Select a completed checkpoint of the requested stage and physical plane")
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-particles", type=int, default=1024)
    parser.add_argument("--exit-particles", type=int, default=4096)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--frame-stride", type=int, default=64)
    parser.add_argument("--width-nm", type=float, default=4.)
    parser.add_argument("--maximum-packets", type=int, default=262144)
    parser.add_argument("--resume-incident-manifest", type=Path,
        help="Reuse an explicitly selected, checksum-verified ideal entrance from identical inputs")
    parser.add_argument("--resume-exit-manifest", type=Path,
        help="Continue the completed Si exit; also select its original incident manifest")
    parser.add_argument("--exit-window-nm", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.source_particles < 1000:
        raise ValueError("At least 1000 original positive-current particles are required")
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    if args.resume_exit_manifest is not None and args.resume_incident_manifest is None:
        raise ValueError("Exit continuation also requires its original incident manifest")
    from temsim.physics.compute_backend import require_coherent_cuda
    require_coherent_cuda("Require GPU")
    import cupy as cp
    from temsim.optics.column import default_state
    from temsim.physics.physical_scene import capture_physical_scene
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
    from temsim.physics.potential_geometry import PotentialGeometrySnapshot
    from temsim.physics.unified_field_runtime import UnifiedFieldRuntime
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.wave_flux import BeamState, WaveMode
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
    from temsim.physics.tip_wave_pipeline import TipWaveRequest
    from temsim.physics.wave_execution import InelasticWaveNumerics
    from temsim.physics.potential_ray import trace_potential_rays
    state = default_state()
    state.simulation_mode = "unified"
    state.acceleration_backend = "Auto"
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for item in collection:
            item.enabled = False
    state.ac_deflector.enabled = state.descan_deflector.enabled = False
    state.beam_blanked = False
    state.nanopulser.installed = False
    state.energy_filter.enabled = False
    state.apertures = []
    state.step_mm = .0002
    sample = state.sample
    sample.specimen_mode, sample.inserted = "atomic", True
    sample.cif_path = str(Path(__file__).resolve().parents[1]/"tests/fixtures/cif/Si.cif")
    sample.thickness_nm = .4
    sample.wave_slice_thickness_angstrom = 2.
    sample.wave_grid_pixels = args.pixels
    sample.wave_field_of_view_angstrom = args.width_nm*10.
    sample.wave_frozen_phonon_enabled = False
    # The benchmark owns no fitted inelastic constants. Missing custom-CIF
    # parameters remain explicitly inactive in the existing material model.
    entrance = sample.z_mm-.5e-6*sample.thickness_nm
    start = entrance-1e-6  # one nanometre of semiclassical propagation
    stop = sample.z_mm+.001
    camera = next(p for p in state.recording_planes if p.key == "camera")
    for plane in state.recording_planes:
        plane.inserted = plane is camera
    camera.z_mm = stop
    scene = capture_physical_scene(state)
    snapshot = ElectromagneticPotentialSnapshot(scene, ())
    geometry = PotentialGeometrySnapshot(start*.001, stop*.001, (), source_scene_geometry_digest=scene.geometry_digest)
    runtime = UnifiedFieldRuntime(snapshot, geometry, capture_instrument_snapshot(state).digest,
        {"scope": "Declared field-free ideal development column; physical instrument fields are not substituted in production"})
    pixels, width, sigma, energy, mass = args.pixels, args.width_nm*1e-9, .2e-9, 300000., .4
    spacing = width/pixels
    coordinate = (np.arange(pixels)-pixels//2)*spacing
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    wave = np.exp(-(xx*xx+yy*yy)/(4*sigma*sigma)).astype(complex)
    wave /= np.linalg.norm(wave)  # definition of this explicitly ideal source
    mode = WaveMode(PlaneWave(wave, np.eye(2)*spacing, np.zeros(2)), mass, TIP_REFERENCE,
        "declared-ideal-single-mode", energy/1000)
    checkpoint = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), start, 1e-9,
        {"schema": "ideal-hybrid-benchmark-source-v1", "physical_feg_phase_chain": False,
         "reference_semantics": "Unit charge accounting only; no physical tip-to-plane equivalence is claimed",
         "sigma_m": sigma, "energy_ev": energy, "source_weight": mass, "seed": args.seed})
    rng = np.random.default_rng(args.seed)
    original_xy = rng.normal(0., sigma, (args.source_particles, 2))
    original_direction_xy = rng.normal(0., hbar/(2*sigma*momentum(energy)), (args.source_particles, 2))
    longitudinal = np.sqrt(1-np.sum(original_direction_xy**2, axis=1))
    original = np.column_stack((original_xy, original_direction_xy/longitudinal[:, None]))
    original_weights = np.full(args.source_particles, mass/args.source_particles)
    traced = trace_potential_rays(snapshot, original, (start*.001, entrance*.001), energy_ev=energy,
        maximum_step_m=1e-9, weights=original_weights)
    wave_request = TipWaveRequest(stop="gun_exit", inelastic=InelasticWaveNumerics(method="zero_loss"))
    wave_request = replace(wave_request, execution=replace(wave_request.execution,
        cache_directory=str(args.output.parent/(args.output.name+"-cache"))))
    numerics = SemiclassicalNumerics(frame_stride=args.frame_stride, window_sigma_pixels=float(args.frame_stride),
        maximum_packets=args.maximum_packets, coefficient_cutoff=1e-10, maximum_step_m=1e-9, packet_batch=512)
    request = HybridRequest("camera", HybridGrid(entrance, pixels, width*1e9),
        source_particles=args.source_particles, exit_particles=args.exit_particles,
        seed=args.seed, semiclassical=numerics, wave=wave_request, exit_window_sigma_nm=args.exit_window_nm)
    request.validate()
    def progress(done, total, message):
        print(f"[{done}/{total}] {message}", flush=True)
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    if args.resume_incident_manifest is None:
        incident = _column(checkpoint, runtime, request, xp=cp, cancelled=lambda: False, progress_callback=progress)
    else:
        incident = restore_ideal_stage(args.resume_incident_manifest, checkpoint, settings,
            "executed-hybrid-column-v1", entrance)
        if incident.record["snapshot_digest"] != snapshot.numerical_digest:
            raise ValueError("Saved ideal entrance belongs to another field snapshot")
        print("Continuing checksum-verified ideal entrance; upstream action was not recalculated", flush=True)
    stage_store = ExecutedWaveStore(wave_request.execution.cache_directory,
        {"scope": "ideal-hybrid-development-intermediate", "source": checkpoint.digest,
         "settings": settings}, 2*1024**3)
    stage_store.put(stage_store.key("specimen-incident"), incident)
    print(json.dumps({"incident_weight": incident.beam.total_weight,
        "frame_reconstruction_l2": incident.record["intervals"][0]["initial_reconstruction_l2"],
        "packet_count": incident.record["intervals"][0]["packet_count"]}), flush=True)
    if args.resume_exit_manifest is None:
        print("Executing real Si material wave", flush=True)
        outgoing = _material(state, incident, runtime, request, cancelled=lambda: False, progress_callback=progress)
    else:
        outgoing = restore_ideal_stage(args.resume_exit_manifest, checkpoint, settings,
            "hybrid-material-exit-v1", sample.z_mm+sample.thickness_nm*.5e-6)
        comparisons = {"incident": outgoing.record["upstream_digest"] == incident.digest,
            "fields": all(row["snapshot_digest"] == snapshot.numerical_digest for row in outgoing.record["material_histories"]),
            "instrument": all(row["input_identity"] == runtime.source_identity for row in outgoing.record["material_histories"])}
        if not all(comparisons.values()):
            raise ValueError(f"Material continuation changed its incident wave, specimen or fields: {comparisons}")
        print("Continuing checksum-verified real Si exit; material wave was not recalculated", flush=True)
    stage_store.put(stage_store.key("specimen-exit"), outgoing)
    print(json.dumps({"material_mode_shapes": [list(mode.plane.amplitude.shape) for mode in outgoing.beam.modes]}), flush=True)
    print("Sampling gauge-covariant local diffraction", flush=True)
    particles, diffraction = sample_exit(outgoing, snapshot, args.exit_particles, numerics, xp=cp, seed=args.seed,
        window_sigma_m=request.exit_window_sigma_nm*1e-9, progress_callback=progress)
    print("Tracing weighted exit particles to the physical receiver", flush=True)
    # The nanometre HK phase step is unnecessary for the field-free classical
    # receiver drift. This explicit numerical change does not affect fields.
    downstream_request = replace(request, semiclassical=replace(numerics, maximum_step_m=2e-7))
    downstream = _downstream(state, runtime, particles, stop, downstream_request)
    detected = downstream["detected_or_absorbed_weight_by_key"].get("camera", 0.)
    report = {"schema": "ideal-hybrid-benchmark-v1", "status": "IDEAL_DEVELOPMENT_VALIDATION",
        "physical_feg_phase_chain": False, "source": {"original_positive_count": args.source_particles,
        "surviving_count": int(traced.alive[-1].sum()), "effective_count": float(original_weights.sum()**2/np.dot(original_weights, original_weights))},
        "source_weight": mass, "incident_weight": incident.beam.total_weight,
        "specimen_exit_weight": outgoing.beam.total_weight, "exit_sampled_weight": float(particles.weight.sum()),
        "detector_weight": detected, "detector_current_a": checkpoint.reference_current_a*detected,
        "downstream_balance_residual": downstream["balance_residual"],
        "coherent_backend": "CUDA", "classical_backend": "CPU",
        "cuda_device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "sample_feedback_on_external_fields": False,
        "settings": settings,
        "qualification": "Ideal single-mode, short field-free column, actual Si CIF; not a full microscope or physical FEG qualification"}
    source_arrays = {"ray_id": np.arange(args.source_particles), "position_m": np.column_stack((original_xy,
        np.full(args.source_particles, start*.001))), "direction": np.column_stack((original_direction_xy, longitudinal)),
        "weight": original_weights, "incident_xy_slopes": traced.xy_slopes[-1]}
    save_result(HybridResult(incident, outgoing, particles, diffraction, downstream, report, source_arrays), args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
