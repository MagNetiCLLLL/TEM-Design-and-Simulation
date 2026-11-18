"""Read executed historical planes only. No historical state is restored or propagated."""

import argparse
import json
from pathlib import Path
from hashlib import sha256

import numpy as np
from temsim.working_point_archive import WorkingPointArchiveIndex
from temsim.particle_section_io import _data_classes, _unpack
from types import SimpleNamespace
from temsim.physics.beam_statistics import original_source_population, transverse_beam_statistics
from trial import write
from measure_probe_waist import observed_minimum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    index = WorkingPointArchiveIndex.read(args.archive)
    package = index.load()
    graph = json.loads(package.arrays[package.metadata["record_graph"]].tobytes())
    fields = graph["mapping"]["simulation"]["fields"]
    observations = {k: fields[k] for k in ("incident", "gun_trace", "incident_checkpoints")}
    sim = SimpleNamespace(**_unpack({"mapping": observations}, package.arrays, _data_classes()))
    cp = sim.incident_checkpoints
    print("Final exact checkpoints", cp.z_mm[-15:], flush=True)
    gun = sim.gun_trace
    population = original_source_population(gun)
    beam = sim.incident
    boundary_dir = args.archive.parent / "validated-particle-boundary"
    checksums = json.loads((boundary_dir / "checksums.json").read_text())
    for name in ("source.json", "source-phase.npz"):
        if sha256((boundary_dir / name).read_bytes()).hexdigest() != checksums[name]:
            raise ValueError("Historical entrance source checksum mismatch")
    boundary = json.loads((boundary_dir / "source.json").read_text())
    if boundary["input_digest"] != package.snapshot.digest:
        raise ValueError("Surface and section were acquired from different inputs")
    entrance = np.load(boundary_dir / "source-phase.npz")
    surface = float(boundary["z_mm"])
    weights = beam.ray_weight
    rows = []
    for i, z in enumerate(beam.z):
        if z < 1522.5 or z > surface:
            continue
        live = (weights > 0) & (beam.alive | (beam.blocked_z > z))
        w = weights[live]
        phase = [getattr(beam, key)[i, live] for key in ("x", "y", "tx", "ty")]
        if not all(np.all(np.isfinite(value)) for value in phase):
            raise ValueError("Nonfinite live observation")
        stats = transverse_beam_statistics(*phase, weights=w)
        rows.append(
            dict(
                z_mm=float(z),
                D95_nm=stats.radius_95_m * 2e9,
                RMS_nm=stats.radius_rms_m * 1e9,
                alpha95_mrad=stats.convergence_95_mrad,
                survivors=int(live.sum()),
                radial_covariance_m_rad=stats.radial_position_angle_covariance_m_rad,
            )
        )
    if rows and rows[-1]["z_mm"] == surface:
        rows.pop()
    if not np.array_equal(entrance["ray_id"], beam.source_ray_id) or not np.array_equal(
        entrance["weight"], weights
    ):
        raise ValueError(
            "The archived stationary reference requires a matched full-survivor population"
        )
    selected = [i for i in (1024, 2048, len(weights)) if i <= len(weights)]
    prefixes = []
    for count in selected:
        mask = (weights > 0) & np.isin(entrance["ray_id"], gun.emission_reference["ray_id"][:count])
        w = weights[mask]
        phase = [entrance[key][mask] for key in ("x_m", "y_m", "tx_rad", "ty_rad")]
        s = transverse_beam_statistics(*phase, weights=w)
        prefixes.append(
            dict(
                original_prefix=count,
                surviving=int(mask.sum()),
                effective=w.sum() ** 2 / (w @ w),
                absolute_subset_current_a=gun.emitted_current_a * w.sum(),
                D95_nm=2e9 * s.radius_95_m,
                rms_radius_nm=1e9 * s.radius_rms_m,
                alpha95_mrad=s.convergence_95_mrad,
                angular_rms_mrad=s.convergence_rms_rad * 1e3,
                radial_covariance_m_rad=s.radial_position_angle_covariance_m_rad,
                core_50pm_current_a=gun.emitted_current_a
                * w[np.hypot(phase[0], phase[1]) <= 50e-12].sum(),
            )
        )
    rows.append(
        dict(
            z_mm=surface,
            D95_nm=prefixes[-1]["D95_nm"],
            RMS_nm=prefixes[-1]["rms_radius_nm"],
            alpha95_mrad=prefixes[-1]["alpha95_mrad"],
            survivors=prefixes[-1]["surviving"],
            radial_covariance_m_rad=prefixes[-1]["radial_covariance_m_rad"],
        )
    )
    z = np.array([r["z_mm"] for r in rows])
    report = dict(
        archive_digest=index.digest,
        input_digest=package.snapshot.digest,
        status="Retrospective stationary ideal-coil archive, scan and descan parked",
        upper_surface_mm=surface,
        population=population,
        emitted_current_a=gun.emitted_current_a,
        nested_prefixes=prefixes,
        through_focus_scope="Retained display nodes from Mini Condenser centre to entrance; no downstream vacuum coverage or sub-node refinement",
        D95=observed_minimum(z, [r["D95_nm"] for r in rows], surface),
        RMS=observed_minimum(z, [r["RMS_nm"] for r in rows], surface),
        retained_planes=rows,
    )
    write(args.output, report)
    print({k: v for k, v in report.items() if k != "retained_planes"})


if __name__ == "__main__":
    main()
