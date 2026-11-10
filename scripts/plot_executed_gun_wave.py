"""Render an existing checksum-verified gun wave; no new coherent calculation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temsim.physics.wave_checkpoint_store import ExecutedWaveStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Do not overwrite a saved result image")
    manifest = json.loads(args.manifest.read_text())
    if manifest["record"]["schema"] != "executed-tip-gun-v1":
        raise ValueError("Select an executed physical gun checkpoint")
    store = ExecutedWaveStore(args.manifest.parent.parent, manifest["dependency"], 192*1024**3)
    checkpoint = store.get(manifest["key"])
    second, first, total, pitch = np.zeros(2), np.zeros(2), 0., 0.
    for mode in checkpoint.beam.modes:
        xy = mode.plane.coordinates_m()*1e6
        mass = abs(mode.plane.amplitude)**2*mode.weight_per_reference_electron
        first += np.sum(xy*mass, axis=(1, 2))
        second += np.sum(xy**2*mass, axis=(1, 2))
        total += float(mass.sum())
        pitch = max(pitch, float(np.max(np.sum(abs(mode.plane.basis_m), axis=1)))*1e6)
    centre = first/total
    sigma = np.sqrt(second/total-centre**2)
    half = float(4*max(sigma))
    bins = max(8, min(256, int(2*half/pitch)))
    edges = [np.linspace(c-half, c+half, bins+1) for c in centre]
    radial_edges = np.linspace(0, half*np.sqrt(2), 257)
    pictures = [np.zeros((bins, bins)), np.zeros((bins, bins))]
    radial = np.zeros(256)
    current_scale = checkpoint.reference_current_a*1e6
    for index, mode in enumerate(checkpoint.beam.modes):
        xy = mode.plane.coordinates_m()*1e6
        mass = abs(mode.plane.amplitude)**2*mode.weight_per_reference_electron
        image, _, _ = np.histogram2d(xy[0].ravel(), xy[1].ravel(), bins=edges,
            weights=mass.ravel()*current_scale)
        pictures[1] += image.T
        if index == 0:
            pictures[0] = image.T
        radius = np.sqrt(np.sum((xy-centre[:, None, None])**2, axis=0))
        radial += np.histogram(radius, bins=radial_edges, weights=mass*current_scale)[0]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.), constrained_layout=True)
    extent = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]
    area = (edges[0][1]-edges[0][0])*(edges[1][1]-edges[1][0])
    limit = max(p.max() for p in pictures)/area
    for ax, picture, title in zip(axes[:2], pictures, ("First executed mode", "Incoherent sum of all 48 modes")):
        im = ax.imshow(picture/area, origin="lower", extent=extent, cmap="magma",
            norm=LogNorm(vmin=limit*1e-6, vmax=limit), interpolation="nearest")
        ax.set(title=title, xlabel="Barrel X (um)", ylabel="Barrel Y (um)")
    fig.colorbar(im, ax=axes[:2], label="Current density (uA / um^2)", shrink=.8)
    axes[2].plot(radial_edges[1:], np.cumsum(radial))
    axes[2].axhline(total*current_scale, color="black", linestyle="--", alpha=.5)
    axes[2].set(xlabel="Radius about current centroid (um)", ylabel="Enclosed current (uA)",
        title="Absolute escaping current")
    axes[2].grid(alpha=.2)
    fig.suptitle(f"Executed physical FEG wave at Z={checkpoint.plane_z_mm:g} mm | CUDA | {total*current_scale:.6f} uA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    metadata = dict(checkpoint_digest=checkpoint.digest, plane_z_mm=checkpoint.plane_z_mm,
        mode_count=len(checkpoint.beam.modes), current_a=total*checkpoint.reference_current_a,
        displayed_current_a=float(pictures[1].sum())*1e-6,
        centroid_um=centre.tolist(), rms_xy_um=sigma.tolist(), image_bins=bins,
        scope="Saved gun-exit wave only; not a specimen wave or a detector image")
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
