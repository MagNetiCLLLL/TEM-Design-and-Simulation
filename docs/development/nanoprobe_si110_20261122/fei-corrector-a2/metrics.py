"""Fit geometric A2 and D4 to a fixed physical specimen-plane population."""

from pathlib import Path
import sys

import numpy as np

LEGACY = Path(__file__).resolve().parents[2] / "cescor_direct_probe_20261121"
sys.path.insert(0, str(LEGACY))
from fit_threefold import angular_sample, design, physical_coefficients
from temsim.physics.beam_statistics import transverse_beam_statistics

CORE_RADIUS_UM = 0.050


def fit_terms(basis, names, xy_um, current):
    root = np.sqrt(current)
    matrix = (root[:, None, None] * basis).reshape(-1, basis.shape[-1])
    norms = np.linalg.norm(matrix, axis=0)
    scaled = matrix / norms
    target = (xy_um * root[:, None]).ravel()
    coeff, _, rank, singular = np.linalg.lstsq(scaled, target, rcond=1e-11)
    if rank != scaled.shape[1]:
        raise ValueError("The pupil sampling does not resolve the aberration basis")
    residual = (target - scaled @ coeff).reshape(-1, 2)
    return (
        physical_coefficients(coeff / norms, names),
        float(singular[0] / singular[-1]),
        float(np.sqrt((residual**2).sum() / current.sum())),
    )


def measure(phase, ids, w_uA, *, uncertainty=False):
    weights = 1000 * w_uA
    stats = transverse_beam_statistics(*phase.T, weights=weights)
    sample = angular_sample(phase, ids, weights, stats)
    basis, names = design(sample)
    coeff, condition, residual = fit_terms(basis, names, sample.xy_um, weights)
    centre = np.average(sample.xy_um, axis=0, weights=weights)
    dx, dy = (sample.xy_um - centre).T
    r = np.hypot(dx, dy)
    order = np.argsort(r)
    r95 = r[order][np.searchsorted(np.cumsum(weights[order]), 0.95 * weights.sum())]
    core = r <= CORE_RADIUS_UM
    mid = (r >= CORE_RADIUS_UM) & (r <= r95)
    phasor = np.average(np.exp(3j * np.arctan2(dy[mid], dx[mid])), weights=weights[mid])
    out = dict(
        A2=coeff["A2"],
        D4=coeff["D4"],
        condition=condition,
        RMS_fit_residual_um=residual,
        D95_um=2 * float(r95),
        RMS_um=float(np.sqrt(np.average(r * r, weights=weights))),
        core_radius_um=CORE_RADIUS_UM,
        core_nA=float(weights[core].sum()),
        axis_core_nA=float(weights[np.hypot(*sample.xy_um.T) <= CORE_RADIUS_UM].sum()),
        H3=float(abs(phasor)),
        H3_angle_deg=float(np.rad2deg(np.angle(phasor)) / 3 % 120),
        H3_annulus_um=[CORE_RADIUS_UM, float(r95)],
        H3_annulus_count=int(mid.sum()),
        current_nA=float(weights.sum()),
        arrivals=len(ids),
        centroid_um=centre.tolist(),
        effective_count=float(weights.sum() ** 2 / (weights @ weights)),
        current_weighted_alpha95_mrad=float(stats.convergence_95_mrad),
    )
    for c in (out["A2"], out["D4"]):
        c.pop("rms_displacement_at_30mrad_um")
    if uncertainty:
        # Blocks of original interleaved Sobol IDs span all eight current strata.
        block = ids // 5000
        jack = []
        for k in range(10):
            keep = block != k
            ab, _, _ = fit_terms(basis[keep], names, sample.xy_um[keep], weights[keep])
            jack.append(ab["A2"]["cos_sin_mm"])
        out["A2_delete_one_block_pairs_mm"] = jack
    return out


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_folder", type=Path)
    args = parser.parse_args()
    with np.load(args.trace_folder / "trajectories.npz") as data:
        state = np.column_stack((data["sample_xy_um"] * 1e-6, data["sample_slopes"]))
        result = measure(
            state, data["sample_ids"], data["sample_weights_uA"], uncertainty=True
        )
    print(json.dumps(result, indent=2))
