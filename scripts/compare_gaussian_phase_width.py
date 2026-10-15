"""Compare two explicitly experimental coherent specimen beamlet readouts."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scripts.plot_gaussian_phase_comparison import require_shared_numerics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coarse", type=Path)
    ap.add_argument("fine", type=Path)
    ap.add_argument("--threshold", type=float, default=1e-4)
    ap.add_argument("--comparison-kind", choices=("packet-width", "axial-step", "quadratic-step"),
                    default="packet-width")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--version-audit", type=Path,
                    help="Explicit audit of exact nonnumerical replay/source digest changes")
    args = ap.parse_args()
    comparison_label = {"packet-width": "packet width", "axial-step": "nonlinear axial step",
                        "quadratic-step": "quadratic integration step"}[args.comparison_kind]
    entries = []
    for path in (args.coarse, args.fine):
        with np.load(path, allow_pickle=False) as data:
            entries.append((data["axis_m"], data["complex_amplitude_density"],
                            json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))))
    (x, a, ra), (xf, b, rb) = entries
    if not np.array_equal(x, xf):
        raise ValueError("Comparison requires identical physical XY coordinates")
    for key in ("mode_id", "energy_kev", "mode_weight", "sample_entrance_z_mm"):
        if ra[key] != rb[key]:
            raise ValueError(f"Different coherent physical mode: {key}")
    source_receipts = [json.loads(Path(r["source_receipt"]).read_text(encoding="utf-8")) for r in (ra, rb)]
    for readout in (ra, rb):
        if hashlib.sha256(Path(readout["source_receipt"]).read_bytes()).hexdigest() != readout["source_receipt_sha256"]:
            raise ValueError("Replay receipt changed after this readout was generated")
        packet_path = Path(readout.get("source_packet", ""))
        if not packet_path.is_file() or hashlib.sha256(packet_path.read_bytes()).hexdigest() != readout.get("source_packet_sha256"):
            raise ValueError("Readout is not bound to its current packet bytes; regenerate it")
    for key in ("input_manifest_sha256", "start_z_mm", "stop_z_mm", "lens_percent", "state_file_sha256"):
        if source_receipts[0].get(key) != source_receipts[1].get(key):
            raise ValueError(f"Refinement comparison physical input mismatch: {key}")
    dimension = {"packet-width": "width", "axial-step": "axial", "quadratic-step": "quadratic"}[args.comparison_kind]
    configuration_audit = require_shared_numerics(source_receipts, dimension, args.version_audit)
    ia, ib = abs(a)**2, abs(b)**2
    area = float((x[1]-x[0])**2)
    valid = (ia >= args.threshold*ia.max()) & (ib >= args.threshold*ib.max())
    phase_difference = np.angle(b*np.conj(a))
    w = np.sqrt(ia[valid]*ib[valid])
    metrics = {
        "scope": f"EXPERIMENTAL semiclassical Gaussian {comparison_label} refinement of one fixed coherent energy component",
        "comparison_kind": args.comparison_kind,
        "configuration_audit": configuration_audit,
        "reference": str(args.coarse), "comparison": str(args.fine),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.coarse, args.fine)},
        "mode_id": ra["mode_id"], "energy_kev": ra["energy_kev"], "mode_weight": ra["mode_weight"],
        "sample_entrance_z_mm": ra["sample_entrance_z_mm"],
        "upstream_manifest_sha256": source_receipts[0]["input_manifest_sha256"],
        "relative_complex_l2": float(np.linalg.norm(b-a)/np.linalg.norm(a)),
        "relative_intensity_l1": float(np.sum(abs(ib-ia))/ia.sum()),
        "relative_intensity_l2": float(np.linalg.norm(ib-ia)/np.linalg.norm(ia)),
        "viewport_probability": [float(ia.sum()*area), float(ib.sum()*area)],
        "probability_difference": float((ib.sum()-ia.sum())*area),
        "phase_mask_fraction_of_own_peak": args.threshold,
        "phase_mask_reference_current_fraction": float(ia[valid].sum()/ia.sum()),
        "phase_mask_comparison_current_fraction": float(ib[valid].sum()/ib.sum()),
        "masked_weighted_phase_rms_rad": float(np.sqrt(np.dot(w, phase_difference[valid]**2)/w.sum())),
        "masked_phase_max_rad": float(abs(phase_difference[valid]).max()),
        "normalization_fitted": False, "piston_removed": False,
        "phase_mixture_claim": False,
        "qualification": f"Empirical {comparison_label} refinement only; full-wave error is not certified. Other discretization, readout quadrature and omitted-current evidence remain separate.",
        "readout_records": [ra, rb],
    }
    out = args.output or args.coarse.parent/"specimen-width-comparison"
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    extent = ((x[0]-(x[1]-x[0])/2)*1e9, (x[-1]+(x[1]-x[0])/2)*1e9)*2
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), constrained_layout=True)
    max_i = max(float(ia.max()), float(ib.max()))*1e-18
    common_norm = LogNorm(max_i*1e-5, max_i)
    if args.comparison_kind == "packet-width":
        labels = [f"{int(round(np.sqrt(r['packet_count'])))} x {int(round(np.sqrt(r['packet_count'])))} beamlets" for r in (ra, rb)]
    elif args.comparison_kind == "axial-step":
        labels = [f"{r['maximum_column_step_mm']:g} mm axial step" for r in source_receipts]
    else:
        labels = [f"{r['quadratic_path_integration']['maximum_quadratic_step_mm']:g} mm quadratic step" for r in source_receipts]
    for col, (field, intensity, label) in enumerate(((a, ia, labels[0]), (b, ib, labels[1]))):
        im = axes[0, col].imshow(intensity*1e-18, origin="lower", extent=extent,
                                  norm=common_norm, cmap="magma")
        axes[0, col].set_title(label+"\nIntensity")
        fig.colorbar(im, ax=axes[0, col], label="Conditional mode density (nm$^{-2}$)")
        phase = np.ma.masked_where(intensity < args.threshold*intensity.max(), np.angle(field))
        im = axes[1, col].imshow(phase, origin="lower", extent=extent,
                                  vmin=-np.pi, vmax=np.pi, cmap="twilight")
        axes[1, col].set_facecolor("#e8e8e8")
        axes[1, col].set_title(label+"\nWrapped phase")
        fig.colorbar(im, ax=axes[1, col], label="rad")
    difference = (ib-ia)*1e-18
    scale = float(abs(difference).max())
    im = axes[0, 2].imshow(difference, origin="lower", extent=extent, cmap="RdBu_r", vmin=-scale, vmax=scale)
    axes[0, 2].set_title("Intensity difference\nFine minus coarse")
    fig.colorbar(im, ax=axes[0, 2], label="nm$^{-2}$")
    difference = np.ma.masked_where(~valid, phase_difference)
    scale = max(float(abs(difference).max()), 1e-12)
    im = axes[1, 2].imshow(difference, origin="lower", extent=extent, cmap="RdBu_r", vmin=-scale, vmax=scale)
    axes[1, 2].set_facecolor("#e8e8e8")
    axes[1, 2].set_title("Phase difference\nNo piston removed")
    fig.colorbar(im, ax=axes[1, 2], label="rad")
    for ax in axes.flat:
        ax.set_xlabel("Laboratory X (nm)")
        ax.set_ylabel("Laboratory Y (nm)")
    fig.suptitle("Experimental coherent Gaussian readout at specimen entrance\n"
                 f"Z = {ra['sample_entrance_z_mm']:.7f} mm | E0 = {ra['energy_kev']:.9f} keV | one mode, not mixture phase", fontsize=13)
    caption = (f"Complex L2 difference {metrics['relative_complex_l2']:.3g}; intensity L1 {metrics['relative_intensity_l1']:.3g}; "
               f"bright-mask phase RMS {metrics['masked_weighted_phase_rms_rad']:.3g} rad. "
               f"Viewport probabilities {metrics['viewport_probability'][0]:.6g}, {metrics['viewport_probability'][1]:.6g}.\n"
               f"No output normalization or phase fit. Dark phase pixels are masked. Empirical {comparison_label} convergence does not certify full-wave accuracy.")
    fig.supxlabel(caption, fontsize=9)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in metrics.items() if k != "readout_records"}, indent=2))


if __name__ == "__main__":
    main()
