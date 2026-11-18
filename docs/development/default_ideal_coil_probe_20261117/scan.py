"""Compare parked entrance sources on the historical 5 nm Si[110] stage raster."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import scan_joint_probe_si110 as screen


def load_reference(folder):
    manifest = json.loads((folder / "checksums.json").read_text())
    for name in ("source.json", "source-phase.npz", "original-gun.npz", "emission-reference.npz"):
        if sha256((folder / name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError("Historical acquisition checksum mismatch")
    meta = json.loads((folder / "source.json").read_text())
    with np.load(folder / "source-phase.npz") as data:
        values = {key: data[key] for key in data.files}
    with np.load(folder / "original-gun.npz") as g, np.load(
        folder / "emission-reference.npz"
    ) as emission:
        ids = g["ray_id"]
        if not np.array_equal(ids, emission["ray_id"]) or len(np.unique(ids)) != len(ids):
            raise ValueError("Invalid original source lineage")
        pos = int(np.sum(g["weight"] > 0))
        if pos != meta["original_positive_count"] or meta["reference_current_a"] <= 0:
            raise ValueError("Invalid original source current")
        lookup = dict(zip(ids, g["weight"]))
        if any(lookup[k] != v for k, v in zip(values["ray_id"], values["weight"])):
            raise ValueError("Historical surviving weights differ from the original gun")
    return (
        values,
        np.ones(len(values["weight"]), bool),
        meta["reference_current_a"],
        pos,
        meta["z_mm"],
        meta["input_digest"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (args.source / "source.json").is_file():
        phase, live, current, n, z, digest = load_reference(args.source)
    else:
        report = json.loads((args.source / "metrics.json").read_text())
        with np.load(args.source / "surface.npz") as source:
            phase = {key: source[key] for key in source.files}
        live = phase["alive"] & (phase["weight"] > 0)
        n = report["population"]["positive_current_count"]
        current = float(phase["emitted_current_a"])
        z = float(phase["z_mm"])
        digest = str(phase["input_digest"])
        if (
            digest != report["input_digest"]
            or z != report["specimen_upper_surface_mm"]
            or np.sum(phase["weight"] > 0) != n
            or len(np.unique(phase["ray_id"])) != len(phase["ray_id"])
        ):
            raise ValueError("Native source identity, counts, or specimen entrance differ")
    if n < 1000 or not current > 0:
        raise ValueError("Insufficient original positive-current source")
    xy = np.column_stack((phase["x_m"][live], phase["y_m"][live])) * 1e9
    slopes = np.column_stack((phase["tx_rad"][live], phase["ty_rad"][live]))
    energy = phase["kinetic_energy_ev"][live]
    absolute = current * phase["weight"][live]
    if not all(np.isfinite(a).all() for a in (xy, slopes, energy, absolute)) or not np.all(
        absolute > 0
    ):
        raise ValueError("Invalid physical entrance phase")
    mu = 1 / np.sqrt(1 + np.sum(slopes**2, axis=1))
    delta = np.array([screen.screened_rutherford_parameter(14, e) for e in energy])
    area = np.array([screen.screened_rutherford_total_cross_section_cm2(14, e) * 1e14 for e in energy])
    repeat, basis, sites = screen.lattice()
    coords = (np.arange(screen.PIXELS) - (screen.PIXELS - 1) / 2) * screen.PITCH
    positions = np.stack(np.meshgrid(coords, coords), axis=-1).reshape(-1, 2)
    sigma = tomllib.loads(screen.PRESET.read_text())["atomistic"]["thermal_sigma_angstrom"] * 0.1
    depth = screen.ray_depths(positions, xy, slopes, area, sites, repeat, sigma, screen.THICKNESS)
    events = max(2, int(poisson.isf(1e-14, depth.max())))
    table = screen.event_responses(mu, delta, events, 768, 1536)
    coarse = screen.event_responses(mu, delta, events, 512, 1024)
    raw, _ = screen.collect(depth, absolute, table)
    old, _ = screen.collect(depth, absolute, coarse)
    vacuum, _ = screen.collect(np.zeros((1, len(absolute))), absolute, table)
    maps = raw.T.reshape(4, len(coords), len(coords))
    levels = []
    for count in sorted(set(k for k in (1024, 2048, n) if k <= n)):
        # Post-hoc nested sums use exactly the original per-particle currents.
        keep = phase["ray_id"][live] < count
        prefix = (
            maps
            if count == n
            else screen.collect(depth[:, keep], absolute[keep], table[:, keep, :])[0].T.reshape(
                maps.shape
            )
        )
        profile_dir = out / f"prefix-{count}"
        profile_dir.mkdir(exist_ok=True)
        profile = screen.central_line(profile_dir, coords, prefix, basis, repeat)
        levels.append(
            dict(source_prefix=count, current_a=float(absolute[keep].sum()), registered_line=profile)
        )
    report = dict(
        status="Fixed-source classical elastic stage-raster comparison",
        input_digest=digest,
        entrance_mm=z,
        original_positive=n,
        surviving=len(absolute),
        effective=absolute.sum() ** 2 / (absolute @ absolute),
        incident_current_a=float(absolute.sum()),
        pixel_pitch_nm=screen.PITCH,
        thickness_nm=screen.THICKNESS,
        raster_shape=[len(coords), len(coords)],
        annulus_mrad=[50, 150],
        bf_mrad=[0, 15],
        vacuum_currents_a=vacuum[0].tolist(),
        angular_refinement_max_a=float(abs(raw - old).max()),
        maximum_current_closure_a=float(abs(raw.sum(axis=1) - absolute.sum()).max()),
        nested_profiles=levels,
        cif_sha256=sha256(screen.CIF.read_bytes()).hexdigest(),
        numerical_model_sha256=sha256(Path(screen.__file__).read_bytes()).hexdigest(),
        scope="Laterally periodic, fixed entrance, ideal exit-angle windows. Long tails illuminate silicon. No dynamic scan-coil or coherent propagation.",
    )
    np.savez_compressed(out / "scan.npz", coords_nm=coords, channel_current_a=maps)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
