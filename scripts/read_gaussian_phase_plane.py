"""Bounded direct readout of experimental coherent Gaussian specimen packets.

Never normalizes the output, fits a phase piston, or claims mixture phase.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["NUMBA_DISABLE_CUDA"] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import ndtr
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scripts.experimental_gaussian_beamlets import GaussianBeamlets


def load_packets(path):
    with np.load(path, allow_pickle=False) as values:
        return GaussianBeamlets(**{key: values[key] for key in values.files})


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def viewport_l2_bound(packets, half_width):
    """Triangle inequality and Gaussian marginal union bound; no incoherent sum."""
    k = 2*np.pi/packets.wavelength_m
    sigma = np.sqrt(np.diagonal(np.linalg.inv(packets.precision_m1.imag)/(2*k), axis1=1, axis2=2))
    q = packets.centres_m
    probability_union = np.sum(ndtr((-half_width-q)/sigma) + ndtr((q-half_width)/sigma), axis=1)
    return float(np.dot(abs(packets.coefficients), np.sqrt(np.minimum(1., probability_union))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("packets", type=Path)
    parser.add_argument("--width-nm", type=float, default=2.)
    parser.add_argument("--pixels", type=int, default=160)
    parser.add_argument("--tag", default="core160")
    parser.add_argument("--method", choices=("direct", "recurrence"), default="direct")
    args = parser.parse_args()
    receipt_path = args.packets.with_name(args.packets.name.replace("-packets.npz", "-receipt.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_sha = sha_file(receipt_path)
    packet_sha = sha_file(args.packets)
    p = load_packets(args.packets)
    width = args.width_nm*1e-9
    dx = width/args.pixels
    axis = (np.arange(args.pixels)-(args.pixels-1)/2)*dx
    start = time.perf_counter()
    field, evidence = p.evaluate_grid(axis, axis, maximum_working_bytes=256*1024**2, method=args.method)
    elapsed = time.perf_counter()-start
    intensity = abs(field)**2
    probability = float(intensity.sum()*dx*dx)
    peak_index = np.unravel_index(np.argmax(intensity), intensity.shape)
    # A conservative packet-wise phase gradient over the requested viewport.
    c = p.precision_m1.real
    gradient = abs(p.momenta-np.einsum("bij,bj->bi", c, p.centres_m)) + np.sum(abs(c), axis=2)*(width/2)
    max_frequency = float(gradient.max()/p.wavelength_m)
    if sha_file(args.packets) != packet_sha or sha_file(receipt_path) != receipt_sha:
        raise ValueError("Packet or replay receipt changed during the readout")
    result = {
        "scope": "EXPERIMENTAL semiclassical Gaussian coherent component; not a mixture phase or full-wave solution",
        "source_receipt": str(receipt_path), "source_receipt_sha256": receipt_sha,
        "source_packet": str(args.packets), "source_packet_sha256": packet_sha,
        "mode_id": receipt["mode_id"], "energy_kev": receipt["energy_kev"], "mode_weight": receipt["weight"],
        "sample_entrance_z_mm": receipt["stop_z_mm"], "packet_count": p.count,
        "width_nm": args.width_nm, "pixels": args.pixels, "pitch_nm": dx*1e9,
        "viewport_probability_midpoint_quadrature": probability,
        "weighted_viewport_probability": probability*receipt["weight"],
        "viewport_omitted_coherent_l2_upper_bound": viewport_l2_bound(p, width/2),
        "peak_density_per_nm2": float(intensity.max()*1e-18),
        "peak_xy_nm": [float(axis[peak_index[1]]*1e9), float(axis[peak_index[0]]*1e9)],
        "conservative_max_packet_phase_cycles_per_pixel": max_frequency*dx,
        "quadrature_convergence": "NOT_YET_COMPARED", "normalization_fitted": False, "piston_removed": False,
        "evaluation": evidence, "elapsed_s": elapsed,
        "initial_reconstruction_relative_complex_l2": receipt["initial_reconstruction"]["relative_complex_l2"],
        "accumulated_screen_triangle_l2_bound": receipt["triangle_l2_bound"],
        "screen_bound_warning": "A bound >=1 is vacuous; width convergence is empirical, not full-wave error certification",
    }
    output = args.packets.with_name(args.packets.stem.replace("-packets", "")+"-"+args.tag)
    np.savez(output.with_suffix(".npz"), axis_m=axis, complex_amplitude_density=field)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
